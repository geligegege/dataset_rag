import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.testing.pickleable import Parent

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_done_task, add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

def step_1_get_content(state: ImportGraphState) -> Tuple[str, str]:
    """
    步骤1：获取Markdown内容
    1. 从state中获取md_path，读取Markdown文件内容。
    2. 如果md_content已经存在（可能是之前节点生成的），直接返回，避免重复读取。
    3. 返回Markdown内容和文件标题（如果有的话）。
    """
    #读取Markdown内容
    md_content = state.get("md_content")
    if not md_content:
        logger.error("Markdown内容不存在，无法进行切块。请确保前置节点已正确生成md_content。")
        raise ValueError("Markdown内容不存在，无法进行切块。")
    #处理md_content中的换行符号，统一为\n，避免不同系统的换行符导致切块问题
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "default_title")
    return md_content, file_title

def step_2_split_by_title(md_content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    '''
    语义切割
    根据标题进行切割，保证语义完整性
    :param md_content: Markdown内容
    :param file_title: 文件标题
    :return: 切割后的块列表，每个块包含content、title、file_title等元数据；
    '''
    #正则
    title_pattern = r'^\s*#{1,6}\s+.+'
    '''
    正则表达式解析：
    ^：匹配行首。
    \s*：匹配行首的任意空白字符（包括空格、制表符等），允许标题前有空格。
    #{1,6}：匹配1到6个连续的#字符，表示Markdown中的标题级别。
    \s+：匹配至少一个空白字符，确保标题符号和标题文本之间有空格分隔。
    .+：匹配标题文本的内容，至少一个字符。
    '''    
    lines=md_content.split("\n")
    #临时存储变量 current_title=str| current_lines=[]| title_count=0 存储了多少块
    #is_code_block=False 用于判断是否在代码块内，代码块内的标题不进行切割
    current_title=""
    title_count=0
    is_code_block=False
    #最终存储的列表
    section=[]
    current_lines=[]
    for line in lines:
        #判断代码块的开始和结束，代码块内不进行标题切割
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            is_code_block=not is_code_block
        if not is_code_block and re.match(title_pattern, line):
            #如果当前行是标题，并且不在代码块内，说明遇到了新的标题
            if current_title or current_lines:
                #如果当前已经有标题或者内容了，说明之前的块已经完成了，可以存储了
                section.append({
                    "title":current_title,
                    "content":"\n".join(current_lines),
                    "file_title":file_title
                    })
                title_count+=1
            #更新当前标题和内容
            current_title=line.strip()
            current_lines=[current_title] #把标题也算到内容里，保证切块的完整性
        else:
            #如果不是标题，继续添加到当前块的内容中
            current_lines.append(line)
    if current_title or current_lines:
        #处理最后一个块，循环结束后如果还有内容没有存储，说明最后一个块还没有存储，需要存储一下
        section.append({
            "title":current_title,
            "content":"\n".join(current_lines),
            "file_title":file_title
            })
        title_count+=1
    logger.info(f"根据标题切割完成，切割成 {title_count} 块，原文共 {len(lines)} 行")
    return section,title_count,len(lines)  

def split_long_section(section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
    '''
    对超过最大长度的块进行二次切分
    :param section: 包含content、title、file_title等元数据的块
    :param max_content_length: 大块切分阈值
    :return: 切分后的块列表，每个块包含content、title、file_title等元数据
    '''
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=max_content_length,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
    sub_sections=[]
    for index,chunk in enumerate(text_splitter.split_text(section["content"]),start=1):
        sub_sections.append({
            "title":f"{section['title']}_{index}",
            "content":chunk.strip(),
            "parent_title":section["title"],
            "part":index,
            "file_title":section["file_title"]
        })
    return sub_sections    

def merge_short_sections(sections: List[Dict[str, Any]], min_length: int) -> List[Dict[str, Any]]:
    '''
    合并小块，按照父标题进行合并，同一父标题下的小块如果长度小于最小长度，就进行合并
    :param sections: 包含content、title、file_title等元数据的块列表
    :param min_length: 短块合并阈值
    :return: 合并后的块列表，每个块包含content、title、file_title等元数据
    '''
    merged_sections=[]
    temp_section=None

    for section in sections:
        if len(section["content"])<min_length:
            #如果当前块的内容长度小于最小长度，说明是一个短块，需要进行合并
            if temp_section is None:
                #如果临时块不存在，说明这是第一个短块，直接赋值给临时块
                temp_section=section
                continue
            is_temp_short=len(temp_section["content"])<min_length
            temp_parent_title = temp_section.get("parent_title", temp_section.get("title"))
            section_parent_title = section.get("parent_title", section.get("title"))

            is_same_parent_title = temp_parent_title == section_parent_title

            if is_temp_short and is_same_parent_title:
                #如果临时块也是短块，并且和当前块的父标题相同，说明可以进行合并
                temp_section["content"]+=f"\n{section['content']}"
                temp_section["part"]=section['part']
            else:
                #如果临时块不是短块，或者和当前块的父标题不同，说明不能进行合并，需要把临时块存储到结果列表中，然后更新临时块为当前块
                merged_sections.append(temp_section)
                temp_section=section
    if temp_section:
        #处理最后一个块，循环结束后如果还有临时块没有存储，说明最后一个块还没有存储，需要存储一下
        merged_sections.append(temp_section)            
    return merged_sections






def step_3_refine_split(sections: List[Dict[str, Any]], min_content_length: int, max_content_length: int) -> List[Dict[str, Any]]:
    '''
    步骤3：细粒度切割和合并
    1.超过最大长度的块进行二次切分
    2.小于最小长度的块进行合并
    :param sections: 步骤2切割后的块列表，每个块包含content、title、file_title等元数据
    :param min_content_length: 短块合并阈值
    :param max_content_length: 大块切分阈值
    :return: 细粒度切割和合并后的块列表
    ''' 
    final_sections=[]
    for section in sections:
        content=section["content"]
        if len(content)>max_content_length:
            #如果块的内容长度超过最大长度，进行二次切分
            sub_section=split_long_section(section, max_content_length)
            final_sections.extend(sub_section)
        else:
            final_sections.append(section)
    #合并小块，按照父标题进行合并，同一父标题下的小块如果长度小于最小长度，就进行合并
    short_sections=merge_short_sections(final_sections, min_content_length)
    #补全元数据，确保每个块都有完整的元数据，方便后续处理
    for section in short_sections:
        section["parent_title"] = section.get("parent_title", section.get("title", ""))
        section["part"] = section.get("part", 1)
    logger.info(f"细粒度切割和合并完成")
    return short_sections

def step_4_backup_chunks(state: ImportGraphState, chunks: List[Dict[str, Any]]) -> None:
    '''
    步骤4：数据备份和状态更新
    1. 将切割后的块列表备份到本地文件（chunks.json），方便调试和后续处理。
    2. 更新state中的chunks属性，存储切割后的块列表，供后续节点使用。
    :param state: 当前的导入图状态对象
    :param chunks: 切割后的块列表，每个块包含content、title、file_title等元数据
    '''
    #备份到本地文件，文件名可以根据任务ID命名，方便区分不同任务的切割结果
    local_dir=state["local_dir"]
    backup_file_path=os.path.join(local_dir, "chunks.json")
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, #将数据写到指定的文件夹
                  f, 
                  ensure_ascii=False, 
                  indent=4
                  )
    logger.info(f"切割后的块列表已备份到本地文件: {backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    '''
    完成md内容的切块！
    最终：chunk->存储块的集合 chunks->备份到本地->chunks.json
    1、参数校验（材料是否完整）
    2、粗粒度切割，使用标题切割保证语义完整性
    3、没有标题的文档给上默认标题
    4、细粒度切割 合适的大小和重叠，大的段落进行二次切分，小的合并
    5、数据的备份和chunks属性的修改
    返回state
    '''
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}")
    add_running_task(state["task_id"], function_name)

    try:
        #1、参数校验（材料是否完整）
        md_content,file_title=step_1_get_content(state)
        #2、粗粒度切割，使用标题切割保证语义完整性
        #[{content:标题的内容，title:标题，file_title:文件名},{}]
        sections,title_count,lines_count=step_2_split_by_title(md_content,file_title)
        # 3、没有标题的文档给上默认标题
        if title_count==0:
            sections=[{
                "title":"no_title",
                "content":md_content,
                "file_title":file_title
            }]
            title_count=1
        #4、细粒度切割 合适的大小和重叠，大的段落进行二次切分，小的合并
        sections=step_3_refine_split(sections,MIN_CONTENT_LENGTH,DEFAULT_MAX_CONTENT_LENGTH)
        #5、数据的备份和chunks属性的修改
        state["chunks"]=sections
        step_4_backup_chunks(state, sections)



    except Exception as e:
        #处理异常
        logger.error(f"执行节点 {function_name} mineru解析发生异常: {e}")
        raise

    finally:
        #6、结束日志和任务状态的配置
        logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
        add_done_task(state["task_id"], function_name)

    return state            

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")    