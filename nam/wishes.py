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
        self.embeddings_model = OpenAIEmbeddings()
        self.vectorized_values: List[List[float]] = self.embeddings_model.embed_documents(values)
        self.vectorized_dict = OrderedDict(zip(self.values, self.vectorized_values))
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )


    def compute_similarity(self, query: str):
        """
        NOT FINISHED!!!
        :param query:
        :return:
        """
        from openai.embeddings_utils import get_embedding, cosine_similarity

        value = self.embeddings_model.embed_query(query)
        #seq(self.vectorized_dict.items()).min_by(lambda emb)
        #cosine_similarity()

        # Initialize the OpenAIEmbeddings class
        embeddings_model = OpenAIEmbeddings()
        pass


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


class ClassifyGenie:

    dropdowns: pl.DataFrame
    prompt: PromptTemplate
    output_parser: StructuredOutputParser
    chain: LLMChain
    field_names: List[str]
    fields: OrderedDict[str, List[str]]


    classify_template = """
    You are molecular biologist with medical background and expertise in non-animal biological models.
    Your task is to read the academic article text in triple quotes and classify it according to the following instructions.
    {format_instructions}
    The text is:
    ```{text}```
    """

    confidence_schema = ResponseSchema(name="confidence", description="You should evaluate how confident you are in the answer on the range from 0 to 100 percents")

    def list_to_string(self, values: List[str]):
        upd = [v.replace(" / ", " or ") for v in values]
        return "[ "+ ", ".join(upd) + " ]"


    def make_field_schema(self, field: str, values: List[str]):
        field_name = field.replace(" / ", " or ").replace(" ", "_")
        additional = self.clarifications[field] if field in self.clarifications else ""
        schemas = [self.make_field_schema(field, value) for field, value in self.fields.items()]
        self.output_parser = StructuredOutputParser.from_response_schemas(schemas + [self.confidence_schema])
        self.prompt = PromptTemplate.from_template(template=self.classify_template, output_parser=self.output_parser)
        description = f"You should assign {field.replace('_', ' ')} for this research text from the following comma-separated list: {self.list_to_string(values)} . Use only the values from the list, do not choose any other ones. {additional}"
        return ResponseSchema(name=field_name, description = description)


    def __init__(self, fields: OrderedDict[str, List[str]], clarifications: dict[str, str] = None, model_name: str = "gpt-3.5-turbo-16k"):
        self.fields = fields
        self.field_names = list(fields.keys()) + ["confidence"]
        self.clarifications: Optional[dict[str, str]] = clarifications
        schemas = [self.make_field_schema(field, value) for field, value in fields.items()]
        self.output_parser = StructuredOutputParser.from_response_schemas(schemas + [self.confidence_schema])
        self.prompt = PromptTemplate.from_template(template=self.classify_template, output_parser=self.output_parser)
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    def predict(self, text: str) -> dict:
        return self.chain.predict(text = text, format_instructions = self.output_parser.get_format_instructions())

    def apply(self, df: pl.DataFrame):
        extra_cols = [pl.col("text").apply(lambda t: self.predict(t)[f]).alias(f+"_predicted") for f in self.field_names]
        return df.with_columns(extra_cols)
