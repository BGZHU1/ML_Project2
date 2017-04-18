import numpy
from sklearn import datasets

#for image one:
#x is the image
#y is the target in excel

class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):

        self.x = input
        self.y = label
        self.w = numpy.zeros((n_in, n_out))



    def sigmoid(self,x):
        #print(x)
        e = numpy.exp(x-numpy.max(x))  # prevent overflow

        return e / numpy.transpose(numpy.array([numpy.sum(e)])) # ndim = 2

    def cross_entropy(self,x):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)

        #sumOfInner=numpy.sum(self.y * numpy.log(self.sigmoid(x).T))

        #cross_entropy=None
        cross_entropy = -numpy.mean(numpy.sum(self.y * numpy.log(self.sigmoid(x).T)
        +(1 - self.y)* numpy.log(1 - self.sigmoid(x).T)))
        print("end of cross entropy")
        return cross_entropy

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        #if input is not None:
            #print("input",input)
            #self.x = input
            #print(self.x)

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        #print(self.x)
        #print(self.W)
        #print(numpy.dot(self.x, self.W) + self.b)
        #print(self.b)
        #muliply two arrays X(input) W(weights) and add b(interception)
        #numpy.dot :For N dimensions it is a sum product over the last axis of a and the second-to-last of b:

        prob_y_x = self.sigmoid(numpy.dot(self.x, self.w))

        d_y = self.y-prob_y_x




        #print(numpy.shape(self.x))
        #print(numpy.shape(d_y))
        self.w += lr * numpy.dot(self.x.T,d_y)
        print("passed !")
        #- lr * L2_reg * self.W

        #print(self.b)
        # cost = self.negative_log_likelihood()
        # return cost



    def compare_result(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        #bias=numpy.random.random()
        result=self.sigmoid(self.w)
        return result


def test_lr(learning_rate=0.01, literation=1):
    # training data

    #iris=datasets.load_iris()
    #print(iris.target)
    #print(iris.data)
    #x=iris.data
    #y=iris.target

    handWritting=datasets.load_digits()
    print(handWritting)
    x=handWritting.data
    y=handWritting.target
    print(x)
    print(y)



    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=len(x[0]), n_out=len(y))

    # train

    for epoch in range(0,literation):
        #first tranin the model
        classifier.train(lr=learning_rate)
        #then caculate the cost
        cost = classifier.cross_entropy(x)
        #set the learning rate
        learning_rate *= 0.95

    # test
    #x = iris.data

    for i in x:
        print(i)
        print("predict result:" , classifier.compare_result(i))



if __name__ == "__main__":
    test_lr()
