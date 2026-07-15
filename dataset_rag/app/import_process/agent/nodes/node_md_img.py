import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

'''
    主要目标：将md中图片进行单独处理，方便后去模型识别图片的含义
    主要动作：图片->文件服务器->图片网络地址  （上文100）图片（下文100）-》视觉模型-》图片总结
    ---》图片的总结（网络地址）-》state-》md_content==新的内容||md_path==处理后的md地址
    总结技术：
        minio
        视觉模型
    总结步骤：
    1、校验并且获取本次操作的数据
        参数：state -》md_path，md_content
        响应： 1、校验后的md_content 2、md路径对象 3、获取图片的文件夹

    2、识别md中使用过的图片，采取下一步操作（进行图片总结）
        参数：1、md_content 2、图片文件夹
        响应：【（照片名，照片地址，（上文，下文））】

    3、进行图片内容的总结和处理（视觉模型）
        参数：【（照片名，照片地址，（上文，下文））】||md文件名称（提示词中md文件名就是存储images的文件名）
        响应：{图片名：总结，....}

    4、上传图片minio以及更新md的内容
        参数：minio_client||{图片名：总结，....}||【（照片名，照片地址，（上文，下文））
        响应：new_md_content

        state[md_content]=new_md_content

    5、进行数据的最新处理和备份
        参数：new_md_content，原md地址-》新md地址
        响应：新的md地址
        state[md_path]=新的md地址
'''

def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, str]:
    """
    提前内容
    :param state: 
    :return: 
    """
   #1、获取md的地址md_path
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path不能为None，请确保前置节点正确设置了md_path")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"Markdown文件未找到: {md_file_path}")
    
    #2、读取md的内容
    if not state["md_content"]:
        #没有再读取，有证明是pdf转md后文件内容
        with md_path_obj.open("r", encoding="utf-8") as f:
            state["md_content"] = f.read()

    #3、获取图片的文件夹（图片和md在同一目录下的images文件夹中）
    # 注意：这里假设传入md的时候，图片都存放在与Markdown文件同级的images文件夹中       
    images_dir = md_path_obj.parent / "images"
    return state["md_content"], md_path_obj, images_dir

def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描Markdown内容，识别使用过的图片，并提取上下文信息
    :param md_content: Markdown文件的文本内容
    :param images_dir: 图片所在的目录路径,绝对路径
    :return: 包含图片信息的列表，每个元素是一个元组：(图片名, 图片地址, (上文, 下文))
    """
    # 1、创建一个目标集合
    targets = []
    #2、循环读取images中的所有图片，判断是否在md_content中被使用过，若是则提取图片的上下文信息（图片前后各100字）
    for image_file in images_dir.iterdir():
        #遍历每个图片文件，判断是否是图片格式
        if not is_supported_image(image_file):
            logger.warning(f"跳过不支持的文件格式: {image_file.name}")
            continue
        #是图片，我们就在md查询，看是否存在，存在读取对应的上下文即可
        #（上，下文）
        content_data=find_image_in_md_content(md_content,image_file)
        if not content_data:
            logger.warning(f"图片{image_file}未在Markdown内容中找到或没有找到上下文")
            continue
        #存在，我们就把图片名称，图片地址，上下文放入目标集合中
        targets.append((image_file.name,str(image_file),content_data))
    return targets


def find_image_in_md_content(md_content: str, image_file: Path, context_length: int = 100) -> Tuple[str, str]:
    """
    在Markdown内容中查找图片，并提取上下文信息
    :param md_content: Markdown文件的文本内容
    :param image_file: 图片文件的路径对象
    :param context_length: 上下文长度
    :return: 上下文信息的元组 (上文, 下文)，如果未找到返回None
    """
    '''
    .......
    ![image](url)
    .......
    '''
    # 定义正则表达式
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file.name) + r".*?\)")#图片引用格式

    content=None
    # 查询符合位置
    item = next(pattern.finditer(md_content), None)
    if item:
        #获取图片位置
        start, end = item.span()#span获取匹配对象的起始和结束位置
        #提取上下文
        pre_context = md_content[max(0, start - context_length):start]
        post_context = md_content[end:min(end + context_length, len(md_content))]
        content = (pre_context, post_context)
    # 截取位置前后的内容
    if content:
        logger.info(f"图片{image_file.name}在Markdown内容中找到，上下文信息已提取: {content}")
        return content


def step_3_generate_img_summaries(targets: List[Tuple[str, str, Tuple[str, str]]], stem: str) -> Dict[str, str]:
    """
    进行图片内容的总结和处理（视觉模型）
    :param targets: 包含图片信息的列表，每个元素是一个元组：(图片名, 图片地址, (上文, 下文))
    :param stem: Markdown文件的名称（不含路径），用于提示词构造
      
    :return: {图片名.xx: 总结内容,图片名.xx: 总结内容,....，图片名.xx: 总结内容}
    """
    #循环每一张图片，向视觉模型发送请求，获取总结内容
    summaries = {}
    request_times = deque()  # 用于记录API请求时间戳，配合速率限制工具使用
    for image_file, image_path, context in targets:
        #访问限速，避免请求过快被API拒绝
        apply_api_rate_limit(request_times, max_requests=20, window_seconds=60)  # 60秒内最多20次请求
        #解构上下文
        #2、1模型对象
        vm_model=get_llm_client(lm_config.vl_model)
        #2、2构造提示词
        prompt = load_prompt("image_summary", root_folder=stem, image_content=context)

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')#字节转字符
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            #直接放图片地址，视觉模型会自动去读取图片内容进行理解
                            #base64编码的图片内容也可以直接放在这里，视觉模型同样可以识别 jpg->jpeg
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    }
                ]
            }
        ]
        #2、3调用模型进行总结
        response=vm_model.invoke(messages)
        #2、4获取总结结果
        summary=response.content.strip().replace("\n", " ")#去掉换行，保持总结内容的连续性
        summaries[image_file]=summary
    logger.info(f"所有图片的总结内容已生成: {summaries}")
    return summaries

def step_4_update_images_and_replace_md(summaries: Dict[str, str], targets: List[Tuple[str, str, Tuple[str, str]]], md_content: str, stem: str) -> str:
    """
    上传图片到MinIO并更新Markdown内容
    :param summaries: {图片名: 总结内容, ...}
    :param targets: 包含图片信息的列表，每个元素是一个元组：(图片名, 图片地址, (上文, 下文))
    :param md_content: 原Markdown内容
    :param stem: Markdown文件的名称（不含路径），用于构造MinIO存储路径
    :return: 更新后的Markdown内容
    """

    #minio存储结果：桶/upload-images/文件夹/图片.jpg
    minio_client = get_minio_client()
    #1、删除MinIO原有图片（如果有的话），避免重复上传和存储冗余
    #1、1 获取要删除的对象
    #Object对象包含name属性（对象名称）和其他元数据
    object_list=minio_client.list_objects(minio_config.bucket_name, prefix=f"{minio_config.minio_img_dir[1:]}/{stem}/", recursive=True)#去掉第一个"/"
    delete_objects_list = [DeleteObject(obj.object_name) for obj in object_list]
    errors=minio_client.remove_objects(minio_config.bucket_name, delete_objects_list)
    for error in errors:
        logger.error(f"删除MinIO对象失败: {error}")
    logger.info(f"已删除MinIO中{stem}原有的图片对象，准备上传新的图片")

    #2、上传图片到MinIO
    #声明记录图片上传结果的字典
    images_url={}
    #targets: [(图片名, 图片地址, (上文, 下文)), ...]
    for image_file, image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"  # 根据实际图片类型设置正确的Content-Type
            )
            images_url[image_file]=f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"图片{image_file}上传MinIO成功，URL: {images_url[image_file]}")
        except Exception as e:
            logger.error(f"图片{image_file}上传MinIO失败: {str(e)}")
            
    #3、md中图片替换即可
    image_infos={}
    for image_file,summary in summaries.items():
        image_url=images_url.get(image_file)
        if not image_url:
            logger.warning(f"图片{image_file}的MinIO URL未找到，跳过Markdown替换")
            continue
        #构造新的Markdown图片引用格式，附带总结内容作为alt文本
        image_infos[image_file]=(summary,image_url)
    logger.info(f"构造新的Markdown图片引用信息: {image_infos}")
    #循环替换Markdown中的图片引用
    if image_infos:
        for image_file, (summary, image_url) in image_infos.items():
            #正则表达式匹配原有的图片引用格式，进行替换
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(f"![{summary}]({image_url})", md_content)
        logger.info(f"已将Markdown中图片{image_file}的引用替换为新的URL和总结内容")
    return md_content        

def step_5_replace_md_and_save(new_md_content: str, md_path_obj: Path) -> str:
    """
    将更新后的Markdown内容保存到新的文件，并返回新文件的路径
    :param new_md_content: 更新后的Markdown内容
    :param md_path_obj: 原Markdown文件的路径对象
    :return: 新Markdown文件的路径字符串
    """
    #构造新的Markdown文件路径，命名规则：原文件名+_new.md
    new_md_path = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    with new_md_path.open("w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"已将更新后的Markdown内容保存到新文件: {new_md_path}")
    return str(new_md_path)


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    

    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    add_running_task(state["task_id"], function_name)
    # 1、校验并且获取本次操作的数据
    #     参数：state -》md_path，md_content
    #     响应： 1、校验后的md_content 2、md路径对象 3、获取图片的文件夹
    md_content,md_path_obj,images_dir=step_1_get_content(state)

    if not images_dir.exists() :
        logger.warning(f"图片文件夹未找到: {images_dir}，直接返回state")
        return state
    
    # 2、识别md中使用过的图片，采取下一步操作（进行图片总结）
    # 参数：1、md_content 2、图片文件夹
    # 响应：【（照片名，照片地址，（上文，下文））】
    targets=step_2_scan_images(md_content,images_dir)

    # 3、进行图片内容的总结和处理（视觉模型）
    #     参数：【（照片名，照片地址，（上文，下文））】||md文件名称（提示词中md文件名就是存储images的文件名）
    #     响应：{图片名：总结，....}

    summaries=step_3_generate_img_summaries(targets,md_path_obj.stem)
    # 4、上传图片minio以及更新md的内容
    #     参数：minio_client||{图片名：总结，....}||【（照片名，照片地址，（上文，下文））
    #     响应：new_md_content
    #     state[md_content]=new_md_content

    new_md_content=step_4_update_images_and_replace_md(summaries,targets,md_content,md_path_obj.stem)

    # 5、进行数据的最新处理和备份
    #     参数：new_md_content，原md地址-》新md地址
    new_md_file_path=step_5_replace_md_and_save(new_md_content,md_path_obj)

    state["md_content"] = new_md_content
    state["md_path"] = new_md_file_path
    logger.info(f">>> [Stub] 执行节点结束: {sys._getframe().f_code.co_name}")
    add_done_task(state["task_id"], function_name)
    return state



if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join("output/hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")    