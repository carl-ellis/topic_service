#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Copyright (C) 2019 Carl Ellis <carlc75@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX [VOCABULARY_SIZE]

Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This actually creates several files:

* `OUTPUT_PREFIX_wordids.txt.bz2`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation in Matrix Market format
* `OUTPUT_PREFIX_bow.mm.index`: index for `OUTPUT_PREFIX_bow.mm`
* `OUTPUT_PREFIX_bow.mm.metadata.cpickle`: titles of documents
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation in Matix Market format
* `OUTPUT_PREFIX_tfidf.mm.index`: index for `OUTPUT_PREFIX_tfidf.mm`
* `OUTPUT_PREFIX.tfidf_model`: TF-IDF model
* `OUTPUT_PREFIX_corpus.pkl.bz2`: WikiModel
* `OUTPUT_PREFIX_docmap.pkl.bz2`: Dictionary mapping Docid -> DocTitle
* `OUTPUT_PREFIX_lda.model`: LDA Model
* `OUTPUT_PREFIX_index.index`: Similarity Matrix

The output Matrix Market files can then be compressed (e.g., by bzip2) to save
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing tokens that appear in more than 10%% of all documents). Defaults to
100,000.

If you have the `pattern` package installed, this script will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern .

Example:
  python make_data enwiki-latest-pages-articles-multistream.xml.bz2 /mnt/gensimmodels/wiki 400000
"""

import bz2
import logging
import os.path
import pickle
import sys

import gensim
from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel

# Wiki is first scanned for all distinct word types (~7M). The types that
# appear in more than 10% of articles are removed and from the rest, the
# DEFAULT_DICT_SIZE most frequent types are kept.
DEFAULT_DICT_SIZE = 100000

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    if not os.path.isdir(os.path.dirname(outp)):
        raise SystemExit("Error: The output directory does not exist. Create the directory and try again.")

    if len(sys.argv) > 3:
        keep_words = int(sys.argv[3])
    else:
        keep_words = DEFAULT_DICT_SIZE

    wiki = WikiCorpus(inp, lemmatize=True)
    wiki.metadata = True  # Ensure doc id is captured

    # only keep the most frequent words
    wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)

    # Save the document ids to titles as a dictionary -- this will take a long time
    # Also may be unnessesary if metadata works correctly
    docmap = {}
    for index, doc in enumerate(wiki.get_texts()):
        docmap[index] = doc[1][1]
    with bz2.BZ2File('doc_index.pickle.bz2', 'w') as f:
        pickle.dump(docmap, f)

    # save dictionary and bag-of-words (term-document frequency matrix)
    MmCorpus.serialize(outp + '_bow.mm', wiki, progress_cnt=10000, metadata=True)
    wiki.dictionary.save_as_text(outp + '_wordids.txt.bz2')

    # load back the id->word mapping directly from file
    # this seems to save more memory, compared to keeping the wiki.dictionary object from above
    dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')
    wiki.save(outp + '_corpus.pkl.bz2')
    del wiki

    # initialize corpus reader and word->id mapping
    mm = MmCorpus(outp + '_bow.mm')

    # build tfidf, ~50min
    tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
    tfidf.save(outp + '.tfidf_model')

    # save tfidf vectors in matrix market format
    MmCorpus.serialize(outp + '_tfidf.mm', tfidf[mm], progress_cnt=10000)

    # Build LDA model
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=1000, update_every=1,
                                          chunksize=10000, passes=5)
    lda.save(outp + "_lda.model")

    # Build the similarity matric
    index = gensim.similarities.Similarity("wiki_index", lda[mm], lda.num_topics)
    index.save(outp + "_index.index")

    logger.info("finished running %s", program)
