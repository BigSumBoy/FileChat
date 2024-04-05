import importlib
import os
from config import *
from typing import List, Dict, Optional
from pathlib import Path
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
import chardet


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def make_text_splitter(
        splitter_name: str = TEXT_SPLITTER_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        llm_model: str = LLM_MODELS[0],
):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name]['headers_to_split_on']
            text_splitter = langchain.text_splitter.MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on)
        else:

            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module('text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if text_splitter_dict[splitter_name]["source"] == "tiktoken":  # 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
            # elif text_splitter_dict[splitter_name]["source"] == "huggingface":  ## 从huggingface加载
            #     if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "":
            #         config = get_model_worker_config(llm_model)
            #         text_splitter_dict[splitter_name]["tokenizer_name_or_path"] = \
            #             config.get("model_path")
            #
            #     if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "gpt2":
            #         from transformers import GPT2TokenizerFast
            #         from langchain.text_splitter import CharacterTextSplitter
            #         tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            #     else:  ## 字符长度加载
            #         from transformers import AutoTokenizer
            #         tokenizer = AutoTokenizer.from_pretrained(
            #             text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
            #             trust_remote_code=True)
            #     text_splitter = TextSplitter.from_huggingface_tokenizer(
            #         tokenizer=tokenizer,
            #         chunk_size=chunk_size,
            #         chunk_overlap=chunk_overlap
            #     )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # If you use SpacyTextSplitter you can use GPU to do split likes Issue #1287
    # text_splitter._tokenizer.max_length = 37016792
    # text_splitter._tokenizer.prefer_gpu()
    return text_splitter


def get_LoaderClass(file_extension: str):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    """
    根据loader_name和文件路径或内容返回文档加载器。
    """
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader", "FilteredCSVLoader",
                           "RapidOCRDocLoader", "RapidOCRPPTLoader"]:
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            loader_kwargs: Optional[Dict] = None,
    ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        self.filename: str = str(Path(filename).as_posix())
        self.ext: str = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs: dict = loader_kwargs
        self.filepath: str = get_file_path("./", filename)

        self.document_loader_name: str = get_LoaderClass(self.ext)
        self.text_splitter_name: str = TEXT_SPLITTER_NAME
        self.docs: Optional[Document] = None

    def __repr__(self):
        return f"KnowledgeFile(filename={self.filename}, ext={self.ext}, " \
               f"document_loader_name={self.document_loader_name}, " \
               f"text_splitter_name={self.text_splitter_name})"
    def file2docs(self):
        if self.docs is None:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            loader = get_loader(loader_name=self.document_loader_name,
                                file_path=self.filepath,
                                loader_kwargs=self.loader_kwargs)
            self.docs = loader.load()
        return self.docs

    def docs2texts(
            self,
            docs: List[Document] = None,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs()
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(splitter_name=self.text_splitter_name, chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
            self,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        docs = self.file2docs()
        self.splited_docs = self.docs2texts(docs=docs,
                                            chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            text_splitter=text_splitter)
# target : file -> text

# 1.file-on-disk --Loader-> KnowledgeFile
# 2.KnowledgeFile --TextSplitter-> splited_docs
#
if __name__ == '__main__':
    kf = KnowledgeFile(filename="./data/151_新农村建设下东平县小型农田水利工程建设现状和对策.pdf")
    print(kf)