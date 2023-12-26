import os
from dotenv import load_dotenv
import argparse
from modules import utils
import psycopg2
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index.vector_stores import PGVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import download_loader
from llama_index import VectorStoreIndex

# gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-vision-preview, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBED_DIMENSION = 384
INDEX_SUFFIX = '-index'
INDEX_TABLE_PREFIX = 'data_'
# LLM_MODEL = 'gpt-3.5-turbo'
LLM_MODEL = 'gpt-4'
SYSTEM_DB = 'postgres'
VECTOR_DB = 'vector_db'
DOCSRAPTORAI_DB = 'docsraptorai'
RAPTOR_DEFAULT_NAME = 'bob'

logger = utils.get_logger('docsraptorai')

class Raptor():
    embed_model = None
    embed_dimension = None
    llm = None
    service_context = None
    db_system= SYSTEM_DB
    db_vector = VECTOR_DB
    db_docsraptorai = DOCSRAPTORAI_DB
    db_host = None
    db_password = None
    db_port = None
    db_user = None
    db_connect_system = None
    db_connect = None
    db_connect_index = None

    def __init__(self):
        logger.info('init')
        self.init_embeddings()
        self.init_llm()
        self.init_service_context()
        self.init_db_connection()

    def init_embeddings(self):
        logger.info('  embeddings')

        self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        self.embed_dimension = EMBED_DIMENSION

    def init_llm(self):
        logger.info('  init LLM')
        self.llm = OpenAI(model=LLM_MODEL)

    def init_service_context(self):
        logger.info('  Define Service Context')

        self.service_context = ServiceContext.from_defaults(
                llm=self.llm, embed_model=self.embed_model
        )
        logger.info('    Set a global service context to avoid passing it into other objects every time')
        set_global_service_context(self.service_context)

    def init_db_connection(self):
        logger.info('  Initialize Postgres')

        logger.info('    system db connection')
        self.db_host = os.getenv('DB_HOST')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_connect_system = psycopg2.connect(
            dbname=self.db_system,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        self.db_connect_system.autocommit = True
        self.init_db()
        logger.info('    docsraptorai db connection')
        self.db_connect = psycopg2.connect(
            dbname=self.db_docsraptorai,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        self.db_connect.autocommit = True
        logger.info('    index vector db connection')
        self.db_connect_index = psycopg2.connect(
            dbname=self.db_vector,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
        )
        self.db_connect_index.autocommit = True

    def init_db(self):
        logger.info(f'    Checking DB {self.db_docsraptorai}')
        with self.db_connect_system.cursor() as c:
            c.execute(f'select exists(select datname from pg_catalog.pg_database where datname=\'{self.db_docsraptorai}\')')
            docsraptorai_db_exist = c.fetchone()[0]
            if not docsraptorai_db_exist:
                logger.info(f'    Creating DB {self.db_docsraptorai}')
                c.execute(f'CREATE DATABASE {self.db_docsraptorai}')

        if os.getenv('DB_RESET_INDEX') == 'true':
            logger.info(f'    Droping DB {self.db_vector}')
            with self.db_connect_system.cursor() as c:
                c.execute(f'DROP DATABASE IF EXISTS {self.db_vector}')

        logger.info(f'    Checking DB {self.db_vector}')
        with self.db_connect_system.cursor() as c:
            c.execute(f'select exists(select datname from pg_catalog.pg_database where datname=\'{self.db_vector}\')')
            vector_db_exist = c.fetchone()[0]
            if not vector_db_exist:
                logger.info(f'    Creating DB {self.db_vector}')
                c.execute(f'CREATE DATABASE {self.db_vector}')

    def get_vector_store(self, index_name):
        logger.info('Get vector store')

        return PGVectorStore.from_params(
            database=self.db_vector,
            host=self.db_host,
            password=self.db_password,
            port=self.db_port,
            user=self.db_user,
            table_name=index_name,
            embed_dim=self.embed_dimension,
        )

    def get_storage_context(self, vector_store):
        logger.info('Get storage context')

        return StorageContext.from_defaults(vector_store=vector_store)

    def get_index(self, index_name):
        logger.info(f'Load index from stored vectors: {index_name}')
        vector_store = self.get_vector_store(index_name)
        storage_context = self.get_storage_context(vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    def get_raptor(self, raptor_name):
        logger.info(f'Get Raptor: {raptor_name}')
        index_name = self.raptor_index(raptor_name)
        return self.get_index(index_name)

    def index_documents(self, index_name, documents):
        logger.info(f'Index documents in index: {index_name}')
        for document in documents:
            logger.info(f'document: {document}')
            logger.info(f'document id: {document.doc_id}')
            logger.info(f'extra info: {document.extra_info}')
        vector_store = self.get_vector_store(index_name)
        logger.info(f'vector store: {vector_store}')
        storage_context = self.get_storage_context(vector_store)
        logger.info(f'storage contect: {storage_context}')
        logger.info('Index in vector store')
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

    def feed_raptor(self, raptor_name, documents):
        logger.info(f'Feed Raptor: {raptor_name}')
        index_name = self.raptor_index(raptor_name)
        return self.index_documents(index_name, documents)

    def raptor_index(self, raptor_name):
        return f'{raptor_name}{INDEX_SUFFIX}'

    def raptor_table(self, raptor_name):
        return f'{INDEX_TABLE_PREFIX}{self.raptor_index(raptor_name)}'

    def get_documents(self, url):
        logger.info(f'Getting documents from: {url}')
        RemoteReader = download_loader("RemoteReader")
        loader = RemoteReader()
        return loader.load_data(url=url)

    def query(self, index, question):
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        logger.info(f'Reponse: {response.response}')
        logger.info(f'Metadata: {response.metadata}')
        return response

    def list(self):
        logger.info('listing raptors')
        return 'Go chase a raptor!'

    def feed(self, url: str):
        logger.info(f'feeding with: {url}')

        documents = self.get_documents(url)
        self.feed_raptor(RAPTOR_DEFAULT_NAME, documents)
        return 'yumi'

    def ask(self, question: str):
        logger.info(f'asking: {question}')
        index = self.get_raptor(RAPTOR_DEFAULT_NAME)
        response = self.query(index, question)
        logger.info(f'Response: {response}')
        if (response.response == 'Empty Response'):
            return 'Rrrr, feed me first'
        else:
            return response.response

    def kill(self):
        logger.info('reset')
        table = self.raptor_table(RAPTOR_DEFAULT_NAME)
        logger.info(f'dropping table: {table}')
        with self.db_connect_index.cursor() as c:
            c.execute(f'DROP table "{table}"')
        return 'Raptor hunted sir'


raptorai = Raptor()
