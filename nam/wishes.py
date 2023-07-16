from collections import OrderedDict
from typing import List, Optional

from functional import seq
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import polars as pl





class DiseaseClassifier:
    """
    Simple classifier
    """
    prompt: PromptTemplate
    output_parser: StructuredOutputParser
    disease_schema: ResponseSchema
    vectorized_dict:OrderedDict[str, List[float]]
    values: List[str]

    llm: BaseChatModel
    chain: LLMChain


    classify_template = """
    You are molecular biologist with medical background and expertise in cellular, organoid and other non-animal disease models.
    Your task is to read the text of academic research paper given in triple quotes and choose the disease from the following list {values}
    ```{text}```
    {format_instructions}
    """

    confidence_schema = ResponseSchema(name="confidence", description="You should evaluate how confident you are in the answer on the range from 0 to 100 percents")

    def __init__(self, values: List[str], model_name: str = "gpt-3.5-turbo-16k"):
        self.prompt = PromptTemplate.from_template(template=self.classify_template)
        self.values = values
        lst = ", ".join(values)
        description=f"You should classify the text in triple quotes according to which disease area it researchers or targets with the therapy or a model. Only choose the value from the following comma-separated disease area list: {lst}"
        self.disease_schema = ResponseSchema(name="disease_area", description = description)
        schemas = [self.confidence_schema, self.disease_schema]
        self.output_parser = StructuredOutputParser.from_response_schemas(schemas + [self.confidence_schema])
        self.prompt = PromptTemplate.from_template(template=self.classify_template, output_parser=self.output_parser)
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    def predict(self, text: str) -> dict[str, str]:
        """
        predictions on a per example level
        :param text:
        :return:
        """
        result = self.chain.predict(text = text,
                                  format_instructions = self.output_parser.get_format_instructions(),
                                  values = self.values)
        return self.output_parser.parse(result)

    def apply(self, df: pl.DataFrame):
        """
        applies predictions to thw whole dataframe
        :param df:
        :return:
        """
        extra_cols = [
            pl.col("text").apply(lambda text: self.predict(text)["disease_area"]).alias("predicted_disease_area"),
            pl.col("text").apply(lambda text: self.predict(text)["confidence"]).alias("confidence")
        ]
        return df.with_columns(extra_cols)