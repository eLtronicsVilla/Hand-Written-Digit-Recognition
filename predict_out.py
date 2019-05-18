import tensorflow as tf
import sys,os
import PIL.Image
def predictint(imvalue):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('model2.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            keep_prob = tf.placeholder(tf.float32)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")    
            y_pred=graph.get_tensor_by_name("y_pred:0")    
            prediction=tf.argmax(y_pred,1)
            return prediction.eval(feed_dict={x:[imvalue],keep_prob:1.0 }, session=sess)


def imageprepare(argv):
   # im = PIL.Image.open(argv).convert('L')
    im = PIL.Image.open(argv)
    tv = list(im.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv]
    return tva
    #print(tva)



def main(argv):
    FileList = []
    for dirname in os.listdir(argv):
                        path = os.path.join(argv,dirname)
                        for filename in os.listdir(path):

                             if filename.endswith(".jpg"):
                                        FileList.append(os.path.join(argv,dirname,filename))


    for filename in FileList:

            imvalue = imageprepare(filename)
            predint = predictint(imvalue)
            print(".....",filename)
            print (predint[0]) #first value in list
            text_file = open('40.ods','a')
            text_file.write(filename +" ")
            text_file.write('{}'.format(predint[0]) + "\n")
            text_file.close()

if __name__ == "__main__":
    main(sys.argv[1])

     
