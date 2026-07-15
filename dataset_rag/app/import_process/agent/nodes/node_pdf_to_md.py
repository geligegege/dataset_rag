import os
from pathlib import Path
import shutil
import sys
import time

from zipfile import ZipFile
import requests
from sqlalchemy import ext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task,add_done_task
from app.utils.path_util import PROJECT_ROOT
from app.conf.mineru_config import mineru_config

def step_1_validate_paths(state):
    '''
    进行路径校验！local_file_path失效，直接异常处理
    local_dir如果没有传入，默认赋值
    :param state:
    :return:
    '''
    logger.debug(f">>> 执行步骤: step_1_validate_paths, 在md转换PDF之前进行路径校验")
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir", None)
    #常规的非空校验
    if not pdf_path:
        logger.error(f"step_1_validate_paths缺少必要参数: local_file_path")
        raise ValueError("step_1_validate_paths缺少必要参数: local_file_path")
    
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        logger.info(f"step_1_validate_paths没有传入local_dir，使用默认值: {local_dir}")
    pdf_path_obj = Path(pdf_path).expanduser()

    if not pdf_path_obj.is_absolute():
        pdf_path_obj = PROJECT_ROOT / pdf_path_obj
    pdf_path_obj = pdf_path_obj.resolve()
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"step_1_validate_paths参数错误: local_file_path路径不存在: {pdf_path}")
        raise FileNotFoundError(f"step_1_validate_paths参数错误: local_file_path路径不存在: {pdf_path}")
    if not local_dir_obj.exists():
        logger.error(f"step_1_validate_paths local_dir路径不存在，正在创建: {local_dir}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)    
    return pdf_path_obj,local_dir_obj

def step_2_upload_and_poll(pdf_path_obj)->str:
    '''
    调用minorul进行PDF解析，返回下载文件的地址。
    :param pdf_path_obj:上传pdf文件的path对象
    :return:str->下载文件zip的地址
    '''
    #1、申请上传解析的地址
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
    "files": [
            {"name":f"{pdf_path_obj.name}"}
        ],
        "model_version":"vlm"
    }
    response = requests.post(url,headers=header,json=data)
    #结果处理，请求http状态时200，且接口返回的code也是0才算成功，否则都算失败
    if response.status_code != 200 or response.json().get("code") != 0:
        logger.error(f"step_2_upload_and_poll申请mineru上传地址失败，状态码: {response.status_code}, 响应内容: {response.text}")
        raise RuntimeError(f"step_2_upload_and_poll申请mineru上传地址失败，状态码: {response.status_code}, 响应内容: {response.text}")
    upload_url = response.json()['data']['file_urls'][0]#拿到上传地址后，进行文件上传
    batch_id=response.json()['data']['batch_id']#拿到batch_id，进行轮询查询结果

    #2、将文件上传到对应的解析地址
    #使用requests库的put方法直接上传文件流，注意不能直接使用put！原因是电脑开了各种服务器，put的请求头会添加一些额外参数，将文件真的转存到第三方服务器！
    #文件存储服务器检查会比较严格！拒绝错误存储！报错！get post都可以，put不行！
    http_session=requests.Session()
    http_session.trust_env = False #关闭环境变量中的代理设置，避免上传文件时走了代理服务器导致的各种问题

    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response=http_session.put(upload_url, data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"step_2_upload_and_poll上传文件到mineru失败，状态码: {upload_response.status_code}, 响应内容: {upload_response.text}")
            raise RuntimeError(f"step_2_upload_and_poll上传文件到mineru失败，状态码: {upload_response.status_code}, 响应内容: {upload_response.text}")
    except Exception as e:
        logger.error(f"step_2_upload_and_poll上传文件到mineru发生异常: {e}")
        http_session.close()
        raise RuntimeError(f"step_2_upload_and_poll上传文件到mineru发生异常: {e}")

    #3、轮询获取解析结果
    #循环轮询，直到解析完成，拿到下载地址
    #设计轮询机制，等待时间设置为3秒，最大等待时间不超过600秒（600页pdf），直到解析完成或者超时
    pool_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    logger.info(f"step_2_upload_and_poll 开始轮询，batch_id={batch_id}, pool_url={pool_url}")
    poll_interval = 3
    timeout_seconds = 600
    start_time = time.time()

    try:
        while True:
            #3.1 超时判断
            if time.time() - start_time > timeout_seconds:
                logger.error(f"step_2_upload_and_poll轮询超时，解析失败")
                raise RuntimeError(f"step_2_upload_and_poll轮询超时，解析失败")
            #3.2 向指定url获取本次解析的结果（用 http_session，不走代理）
            res = http_session.get(pool_url, headers=header, timeout=30)
            #3.3 解析结果判断和获取zip_url
            if res.status_code != 200:
                #5xx状态码是服务器错误，可能是暂时性的，继续轮询；其他状态码直接报错
                if 500 <= res.status_code < 600:
                    time.sleep(poll_interval)
                    continue
                raise RuntimeError(f"step_2_upload_and_poll轮询获取结果失败，状态码: {res.status_code}, 响应内容: {res.text}")
            res_json=res.json()
            if res_json.get("code") != 0:
                logger.error(f"step_2_upload_and_poll轮询获取结果接口返回错误，状态码: {res.status_code}, 响应内容: {res.text}")
                raise RuntimeError(f"step_2_upload_and_poll轮询获取结果接口返回错误，状态码: {res.status_code}, 响应内容: {res.text}")

            #判断解析状态    
            extract_result=res_json['data']['extract_result'][0]
            if extract_result['state'] == "done":
                zip_url=extract_result['full_zip_url']
                logger.info(f"step_2_upload_and_poll轮询获取结果成功，解析完成，zip_url: {zip_url}，耗时: {time.time() - start_time}秒")
                return zip_url
            else:
                time.sleep(poll_interval)
    except Exception as e:
        logger.error(f"step_2_upload_and_poll轮询获取结果发生异常: {e}")
        raise RuntimeError(f"step_2_upload_and_poll轮询获取结果发生异常: {e}")            
    finally:
        http_session.close()

def step_3_download_and_extract(zip_url,local_dir_obj,file_stem)->str:
    '''
    下载zip文件并解压，返回解压后的md文件路径。
    :param zip_url:下载文件的地址
    :param local_dir_obj:本地存储目录的path对象
    :param file_stem:原文件的文件名（不带后缀），用来命名解压后的md文件
    :return:str->md文件的路径
    '''
    #1、下载zip文件
    try:
        response = requests.get(zip_url)
        if response.status_code != 200:
            logger.error(f"step_3_download_and_extract下载zip文件失败，状态码: {response.status_code}, 响应内容: {response.text}")
            raise RuntimeError(f"step_3_download_and_extract下载zip文件失败，状态码: {response.status_code}, 响应内容: {response.text}")
        
        #2、zip文件保存到本地
        zip_save_path = local_dir_obj / f"{file_stem}.zip"
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"step_3_download_and_extract下载zip文件成功，保存路径: {zip_save_path}")    
        
        #3、将上一次下载的文件夹内容进行删除，避免占用空间
        extract_target_dir = local_dir_obj / file_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
            logger.info(f"step_3_download_and_extract删除上一次的解压文件夹成功，路径: {extract_target_dir}")

        #创建新的目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        #4、进行zip文件的解压
        with ZipFile(zip_save_path, 'r') as zip_ref:
            #调用zip_ref.extractall方法直接解压到目标文件夹，注意目标文件夹必须存在，否则会报错！解压后的文件结构是：extract_target_dir/file_stem.md
            zip_ref.extractall(extract_target_dir)

        #5、返回md文件的地址
        #解压后的文件的文件名可能叫文件.md，也可能叫full.md
        md_file_list=list(extract_target_dir.glob("*.md"))
        target_md_path=None

        if not md_file_list:
            logger.error(f"step_3_download_and_extract解压完成后没有找到md文件，解压目录: {extract_target_dir}")
            raise RuntimeError(f"step_3_download_and_extract解压完成后没有找到md文件，解压目录: {extract_target_dir}")

        #检查有没有文件名.md的文件，如果有就用这个，如果没有就用full.md
        for md_path in md_file_list:
            if md_path.name == f"{file_stem}.md":
                target_md_path=md_path
                break

        if not target_md_path:
            # 如果没有找到匹配的文件，尝试使用full.md
            for md_path in md_file_list:
                if md_path.name.lower() == "full.md":
                    target_md_path=md_path
                    break

        if not target_md_path:
            target_md_path=md_file_list[0]#如果连full.md都没有，就用第一个md文件，虽然不太合理，但总比没有md文件好

        if target_md_path.stem != file_stem:
            #如果最终选定的md文件的文件名和原文件名不一致，就进行重命名，保持和原文件名一致，方便后续处理
            new_md_path=extract_target_dir / f"{file_stem}.md"
            target_md_path.rename(new_md_path)
            target_md_path=new_md_path

        final_md_path=str(target_md_path.resolve())    
        logger.info(f"step_3_download_and_extract解压zip文件成功，md文件路径: {final_md_path}")
        return final_md_path

    except Exception as e:
        logger.error(f"step_3_download_and_extract下载或解压发生异常: {e}")
        raise RuntimeError(f"step_3_download_and_extract下载或解压发生异常: {e}")

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 进入日志和任务状态的配置
    2、进行参数校验（local_dir）给与默认值|local_file_path必须存在且是pdf文件
    3、调用minorul进行PDF解析，返回下载文件的地址。
    4、下载zip文件并解压，解析提取到local_dir目录下。
    5、把md_path地址进行赋值，读md内容到md_content。
    6、结束日志和任务状态的配置
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
    add_running_task(state["task_id"], function_name)

    try:
        #2、进行参数校验（local_dir）给与默认值|local_file_path必须存在且是pdf文件
        pdf_path_obj,local_dir_obj=step_1_validate_paths(state)
        #3、调用minorul进行PDF解析，返回下载文件的地址。
        zip_url=step_2_upload_and_poll(pdf_path_obj)
        #4、下载zip文件并解压，解析提取到local_dir目录下。
        md_path=step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
        state['md_path']=md_path
        state['local_dir']=str(local_dir_obj)
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content']=f.read()

    except Exception as e:
        #处理异常
        logger.error(f"执行节点 {function_name} mineru解析发生异常: {e}")
        raise

    finally:
        #6、结束日志和任务状态的配置
        logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
        add_done_task(state["task_id"], function_name)

    return state            


if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")    