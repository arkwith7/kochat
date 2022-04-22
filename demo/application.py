"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""

# 소스다운 받은 위치를 고려해서 개발환경 구성 할때마다 추가 필요
import sys
sys.path.append('/home/diquest/chatbot/kochat')


from flask import render_template

from kochat.app import KochatApi
from kochat.data import Dataset
from kochat.loss import CRFLoss, CosFace, CenterLoss, COCOLoss, CrossEntropyLoss
from kochat.model import intent, embed, entity
from kochat.proc import DistanceClassifier, GensimEmbedder, EntityRecognizer, SoftmaxClassifier

from demo.scenario import dust, weather, travel, restaurant
# from scenario import dust, weather, travel, restaurant
# 에러 나면 이걸로 실행해보세요!

import time
import datetime

start = time.time()

dataset = Dataset(ood=True)
emb = GensimEmbedder(model=embed.FastText())

clf = DistanceClassifier(
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
)

rcn = EntityRecognizer(
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

kochat = KochatApi(
    dataset=dataset,
    embed_processor=(emb, True),
    intent_classifier=(clf, True),
    entity_recognizer=(rcn, True),
    scenarios=[
        weather, dust, travel, restaurant
    ]
)

end = time.time()

sec = (end - start) 
result_list = str(datetime.timedelta(seconds=sec)).split(".") 
print("초기 서버 시작하는데 소요된 시간 : ",result_list[0])


@kochat.app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    

    kochat.app.template_folder = kochat.root_dir + 'templates'
    kochat.app.static_folder = kochat.root_dir + 'static'
    # kochat.app.run(port=8080, host='0.0.0.0')
    kochat.app.run(port=8080, host='127.0.0.1')


