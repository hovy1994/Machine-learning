
from __future__ import print_function
import grpc
from google.protobuf import json_format
import file_pb2
import file_pb2_grpc
from threading import Thread
import json
from collections.abc import Iterator
import itertools
from json.encoder import JSONEncoder
import ast
from multiprocessing import Queue
import time

from skgarden import MondrianForestRegressor
import numpy as np
import pandas as pd
import datetime
import rrcf

try:
    import queue
except ImportError:
    import Queue as queue


class Client:
    """ gRPC Client class for streaming competition platform"""
    channel = None
    stub = None

    def __init__(self, batch_size):
        """

        :param batch_size: Integer value, defined by the competition and available at competition page
        :param server_port: Connection string ('IP:port')
        :param user_email: String, e-mail used for registering to competition
        :param token: String, received after subscription to a competition
        :param competition_code: String, received after subscription to a competition
        :param first_prediction: Prediction, class generated from .proto file. Used to initiate communication with the
        server. Not influencing the results. Should contain appropriate fields from .proto file.
        """

        # mondrian
        self.mfr = MondrianForestRegressor(random_state=1, n_estimators=100, bootstrap=True)
        self.previous_target_3 = pd.Series()
        self.features_for_rowID = Queue()
        self.previous_train_batch = np.array([-1, -1, -1, -1, -1])
        # rrcf
        self.num_trees = 40
        self.tree_size = 256
        self.forest = []
        self.avg_codisp = {}
        self.curr_sum = 0
        self.curr_num = 0
        self.idx = 0

        self._init_modeling()

        a = 1
        while a == 1:
            print("wait")
            now = datetime.datetime.now()
            starttime = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if now >= starttime:
                print(now)
                print("시작!")
                break

        self.batch_size = batch_size
        self.stop_thread = False
        self.predictions_to_send = Queue()
        self.channel = grpc.insecure_channel('app.streaming-challenge.com:50051')
        self.stub = file_pb2_grpc.DataStreamerStub(self.channel)
        self.user_email = 'hovy199431@gmail.com'
        self.competition_code = 'jR' #oj
        self.token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoieXU5OTA1MjRAZ21haWwuY29tIiwiY29tcGV0aXRpb25faWQiOiIxIn0.B7CAjAsEbTjp4l1K4GR1Y0IJZj6_mKEbKBXsXXJmGBg'
        self.predictions_to_send.put(file_pb2.Prediction(rowID=1000, target=333))
        self.metadata = self.create_metadata(user_id=self.user_email, code=self.competition_code, token=self.token)



    @staticmethod
    def create_metadata(user_id, code, token):
        """
        :param user_id:
        :param code:
        :param token:
        :return:
        """
        metadata = [(b'authorization', bytes(token, 'utf-8')), (b'user_id', bytes(user_id, 'utf-8')),
                    (b'competition_id', bytes(code, 'utf-8'))]
        return metadata

    @staticmethod
    def create_forest(num_trees):

        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)

        return forest

    def partial_train(self, X_test, y_test):
        y_pred, y_std = self.mfr.predict(X_test, return_std=True)
        self.mfr.partial_fit(X_test, y_test)
        #print('pred : %f, std: %f, y: %f'%(y_pred, y_std, y_test))
        return y_pred, y_std

    def _init_modeling(self):
        network = pd.read_csv('initial_training_data.csv', index_col='date', parse_dates=['date'])

        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree()
            self.forest.append(tree)

        train_len = len(network)
        #train_len = 1000
        train_start = 80000
        self.idx = 0

        print("start!")

        for index in range(train_start, train_len):
            point = float(network[index:index + 1].values)  # get one by one

            for tree in self.forest:
                if len(tree.leaves) > self.tree_size:
                    tree.forget_point(self.idx-self.tree_size)

                tree.insert_point(point, index=self.idx)

                if not index in self.avg_codisp:
                    self.avg_codisp[self.idx]=0
                self.avg_codisp[self.idx] += tree.codisp(self.idx) / self.num_trees

            # avg_codisp은 (각 tree 이 point를 anomaly로 생각하는 정도)의 평균
            mean = np.array(list(self.avg_codisp.values())).mean()
            std = np.array(list(self.avg_codisp.values())).std()

            z = (self.avg_codisp[self.idx] - mean)/std
            self.idx += 1

            if z > 3.0 or z < -3.0:
                # if abs(z-score) is over 3.0
                # replace the value with the mean of prev 5 days
                network.iloc[index] = network[index-5 : index].mean() #

        print("init_modeling에서 anomaly detection 완료")


        print("init_modeling에서 trainign 시작")
        for i in range(7+train_start, train_len):
            X_train = pd.Series()
            X_train['prev1'] = float(network[i - 7:i - 6]['target'].values)
            X_train['prev2'] = float(network[i - 6:i - 5]['target'].values)
            X_train['prev3'] = float(network[i - 5:i - 4]['target'].values)
            y_train = (network[i:i + 1]['target'].values)
            self.mfr.partial_fit(X_train.values.reshape(1, -1), y_train)
        print("train 완료")

        self.previous_target_3['prev3'] = float(network[train_len - 8:train_len - 7]['target'].values)
        self.previous_target_3['prev2'] = float(network[train_len - 7:train_len - 6]['target'].values)
        self.previous_target_3['prev1'] = float(network[train_len - 6:train_len - 5]['target'].values)
        self.previous_train_batch = network[train_len-5:train_len]['target'].values

        print('endebded')


    def generate_predictions(self):
        """
        Sending predictions

        :return: Prediction
        """
        while True:
            try:
                prediction = self.predictions_to_send.get(block=True, timeout=60)
                print("Prediction: ", prediction)
                yield prediction
            except queue.Empty:
                self.stop_thread = True
                break



    #check anomaly with RRCF
    def anomaly_detection(self, data):
        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.idx-self.tree_size)
              
            tree.insert_point(data, index=self.idx)

            if not self.idx in self.avg_codisp:
                self.avg_codisp[self.idx] = 0
            self.avg_codisp[self.idx] += tree.codisp(self.idx) / self.num_trees
        # avg_codisp은 (각 tree 이 point를 anomaly로 생각하는 정도)의 평균
        mean = np.array(list(self.avg_codisp.values())).mean()
        std = np.array(list(self.avg_codisp.values())).std()

        z = (self.avg_codisp[self.idx] - mean)/std
        self.idx += 1
        if z > 3.0 or z < -3.0 :
            return self.previous_train_batch.mean()
            # if abs(z-score) is over 3.0
            # replace the value with the mean of whole data we met

        else :
            return data
        #if not over 3.0, then no need to replace the value



    def loop_messages(self):
        """
        Getting messages (data instances) from the stream.

        :return:
        """

        #generate prediction -> get prediction from predictions_to_send one by one ans SEND to server

        messages = self.stub.sendData(self.generate_predictions(), metadata=self.metadata)
        test_idx = 0
        test_feature = self.previous_target_3

        try:
            for message in messages:

                message = json.loads(json_format.MessageToJson(message))
                print("message:", message)
                if message['tag'] == 'TEST':
                    print('test')
                    test_feature['prev3'] = test_feature['prev2']
                    test_feature['prev2'] = test_feature['prev1']
                    test_feature['prev1'] = float(self.previous_train_batch[test_idx])

                    pred = self.mfr.predict(test_feature.values.reshape(1, -1))
                    prediction = file_pb2.Prediction(rowID=message['rowID'], target=pred)
                    self.predictions_to_send.put(prediction)

                    #
                    test_idx =(test_idx+1)%5
                    print(test_idx)

                    print('test end')

                if message['tag'] == 'TRAIN':
                    print('train')
                    #training data to train my model.

                    target = message['target']
                    target = self.anomaly_detection(target)

                    print(self.previous_target_3)

                    # i-5, i-6, i-7 의 값을 갖고 학습
                    if  self.previous_target_3['prev3'] < 0 :
                        self.previous_target_3['prev3'] = target
                    elif self.previous_target_3['prev2'] < 0 :
                        self.previous_target_3['prev2'] = target
                    elif self.previous_target_3['prev1'] < 0 :
                        self.previous_target_3['prev1'] = target
                    else :
                        print('else')
                        #replace the oldest value
                        self.previous_target_3['prev3'] = self.previous_target_3['prev2'] #-7
                        self.previous_target_3['prev2'] = self.previous_target_3['prev1'] #-6
                        self.previous_target_3['prev1'] = float(self.previous_train_batch[0]) #-5

                        # partial fit with 3 previous values as feature
                        self.mfr.partial_fit(self.previous_target_3.values.reshape(1, -1), [target])

                        #현재 train data의 target값 저장
                        self.previous_train_batch = np.roll(self.previous_train_batch, -1)
                        self.previous_train_batch[4] = target

                        print('else end')

                    print('train end')

                if self.stop_thread : break

        except Exception as e:
            print(str(e))
            pass

    def run(self):
        """
        Start thread.
        """
        print("Start")
        t1 = Thread(target=self.loop_messages)
        t1.start()


if __name__ == "__main__":
    client_1 = Client(batch_size=5)
    client_1.run()
