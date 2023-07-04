from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI 
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator 
from threading import Thread 
import time
import sys 

class LoaderThread:

    def run(self):
        print(self.text)
        while self.running:
            time.sleep(1)
            self.t += 1
    
    def start(self, text):
        self.text = text 
        self.running = True
        self.t = 0
        self.thread = Thread(target=self.run)
        self.thread.start()
    
    def stop(self):
        self.running = False 
        self.thread.join()
        print("It took {0} seconds".format(self.t))

class DocProcessor:
    def __init__(self, type = 1):
        if type == 1:
            self.LoaderType = CSVLoader 
        self.lt = LoaderThread()
        
    def load_file(self, file_path):
        self.loader = self.LoaderType(file_path = file_path)
        self.lt.start("Waiting for indexing")
        self.index = VectorstoreIndexCreator(vectorstore_cls = DocArrayInMemorySearch).from_loaders([self.loader])
        self.lt.stop()
    
    def query(self, text):
        self.lt.start("Waiting for index response")
        response = self.index.query(text)
        self.lt.stop()
        return response 

if __name__ == "__main__" and len(sys.argv) >= 3:
    dp = DocProcessor()
    dp.load_file(sys.argv[1])
    print(dp.query(" ".join(sys.argv[2:])))
