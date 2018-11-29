import json
import logging
import os

from aiohttp import web

import gensim
import numpy
import tomodachi


WORDLIST_FILE = os.environ.get("TOPIC_WORDLIST_FILE")
CORPUS_FILE = os.environ.get("TOPIC_CORPUS_FILE")
LDA_FILE = os.environ.get("TOPIC_LDA_FILE")


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumPyEncoder, self).default(obj)


@tomodachi.service
class TopicService(tomodachi.Service):

    id2word = gensim.corpora.Dictionary.load_from_text(WORDLIST_FILE)
    corpus = gensim.corpora.MmCorpus(CORPUS_FILE)
    lda = gensim.models.LdaModel.load(LDA_FILE)

    name = "topic_service"
    log_level = 'INFO'

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
    async def topic(self, request: web.Request) -> str:
        data = await request.json()
        text = data.get("text", "")

        # Process text to BOW
        doc = gensim.utils.simple_tokenize(text)
        bow = TopicService.id2word.doc2bow(list(doc))

        doc_lda = TopicService.lda[bow]

        # Extract Topics
        topics = sorted(doc_lda, key=lambda x: x[1], reverse=True)
        topic_data = [(t[0], t[1], TopicService.lda.show_topic(t[0])) for t in topics]

        return tomodachi.HttpResponse(body=json.dumps({"data": topic_data}, cls=NumPyEncoder), status=200, content_type='application/json')

