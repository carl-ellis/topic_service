import logging
import os

from aiohttp import web

import gensim
import tomodachi


WORDLIST_FILE = os.environ.get("TOPIC_WORDLIST_FILE")
CORPUS_FILE = os.environ.get("TOPIC_CORPUS_FILE")
LDA_FILE = os.environ.get("TOPIC_LDA_FILE")


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


@tomodachi.service
class TopicService(tomodachi.Service):

    id2word = gensim.corpora.Dictionary.load_from_text(WORDLIST_FILE)
    corpus = gensim.corpora.MmCorpus(CORPUS_FILE)
    lda = gensim.models.LdaModel.load(LDA_FILE)

    name = "topic_service"

    options = {
        'http': {
            'port': 4711,
            'content_type': 'application/json',
            'charset': 'utf-8',
        }
    }

    @tomodachi.http("GET", r"/?")
    async def index(self, request: web.Request) -> str:
        return "topic"

    @tomodachi.http("POST", r"/topic/?")
    def topic():
        pass

