



# 1. Сд��
input_str = input_str.lower()

# 2. ɾ������
import re
input_str = re.sub(r'\d+', '', input_str)

# 3. ɾ��������    string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
import string
input_str = input_str.translate(str.maketrans('', '', string.punctuation))

# 4. ɾ�����߿ո�
input_str = input_str.strip()

# 5. ɾ��Stopword
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwords_eng = set(stopwords.words('english'))
input_str = ''.join([x for x in word_tokenize(input_str) if x not in stopwords_eng])

# 6. Stemming����ȡ�ʸ�/�ʸ�  types-->type  stemming-->stem
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()
input_str = ''.join([stemmer.stem(x) for x in word_tokenize(input_str)])

# 7. Lemmatization�����λ�ԭ����ȡ���ʻ�����ʽ  mice-->mouse
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
input_str = ''.join([lemmatizer.lemmatize(x) for x in word_tokenize(input_str)])

# 8. ���Ա�עPOS
from textblob import TextBlob
input_str = 'Parts of speech examples: an article, to write, interesting, easily, and, of'
result = TextBlob(input_str)
print(result.tags)

# 9. �ֿ�(ǳ����)��ʶ����ӵ�˳����ɵ�λ(�����顢����������)
from textblob import TextBlob
input_str='A black television and a white stove were bought for the new apartment of John.'
result = TextBlob(input_str)
rp = nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}')
result = rp.parse(result.tags)
print(result)

# 10. ����ʵ��ʶ��
from nltk import word_tokenize, pos_tag, ne_chunk
input_str = 'Bill works for Apple so he went to Boston for a conference.'
print(ne_chunk(pos_tag(word_tokenize(input_str))))

# 11. �ʴ�����ȡ ICE: Idiom and Collocation Extractor for Research and Education
input = ['he and Chazz duel with all keys on the line.']
from ICE import CollocationExtractor
extractor = CollocationExtractor.with_collocation_pipeline('T1', bing_key = 'Temp', pos_check = False)
print(extractor.get_collocations_of_length(input, length = 3))
