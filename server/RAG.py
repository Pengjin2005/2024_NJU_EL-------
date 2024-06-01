from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

class RAG:
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"

    def __init__(self):
        model_id = "damo/nlp_corom_sentence-embedding_chinese-base"
        self.embeddings = ModelScopeEmbeddings(model_id=model_id)

        self.db = Milvus(self.embeddings, collection_name="poems")

        self.llm = MoonshotChat(
            model="moonshot-v1-128k",
            api_key="sk-iWE3G6vwSONG3hTsHzDJS1h3RFkqxZyiTHiF98AF2MfXAqKQ",
            )
        
        self.prompt_template = PromptTemplate.from_template(
        """
# 描述: 你是一个经验丰富的诗人和文学评论家, 你非常了解中国传统古典文学. 拥有对典型意象, 意境的理解. 你可以润色用户创作的现有诗句; 为斟酌词句提供符合传统意象和具体语境的推荐; 还能够就诗词提供丰富的联想查询功能. 
# 目标: 
1. 你要根据为你提供的"参考资源"进行相关工作.
2. 如果用户给你提供了具体的诗词并要求你进行润色, 你应当根据参考资源给出具体的修改建议, 包括用词, 格律, 意境.
3. 如果用户给你提供了不完全的诗句并要求你给出推荐, 你一定给出符合句子文风, 格律要求, 符合句意的用词推荐.
4. 如果用户给你提供了一段文字并询问你有没有相似的诗句, 你要给出符合对应意境, 情绪的诗句.
# 限制:
1. 请不要过于礼貌, 冷静客观的回答问题就好.
2. 如果没有相关内容, 请不要强行回答, 直接告知用户无法回答此问题.

#参考资料:
{content}
#问题:
{question}
        """
        )
        
    def prompt(self, question: str) -> str:
        results = self.db.similarity_search(question, k=6)
        content = "".join([result.page_content+'\n\n' for result in results])
        prompt = self.prompt_template.format(content=content, question=question)
        print(prompt)
        return prompt
    
    def answer(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    rag = RAG()
    stri = "烟笼寒水夜笼沙。 请创作出下一句。"
    ans = rag.answer(rag.prompt(stri))
    print(ans)