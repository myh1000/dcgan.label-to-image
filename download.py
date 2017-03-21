import urllib2
import urllib
import json
import numpy as np
import cv2
import untangle
import scipy.misc
import tempfile
from preprocess import *

def run(tag_classes):

    count = 0
    fail = 0
    maxsize = 512

    # add glasses + hair color / combos

    # tag_classes = ["blue_hair", "red_hair", "blonde_hair"]
    imgs = []
    tags = []
    tagname = []

    for idx, tag in enumerate(tag_classes):
        print("Now downloading... " + tag)
        for i in xrange(10):
            stringreturn = urllib2.urlopen("http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl+"+tag+"&pid="+str(i+20)).read()
            xmlreturn = untangle.parse(stringreturn)
            for post in xmlreturn.posts.post:
                imgurl = "http:" + post["sample_url"]
                print imgurl
                if ("png" in imgurl) or ("jpg" in imgurl):
                    resp = urllib.urlopen(imgurl)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    scipy.misc.imsave("imgs/"+str(count)+".jpg",image[...,::-1], "jpeg")
                    if process_img("imgs/"+str(count)+".jpg") != 0:
                        imgs.append("imgs/"+str(count)+".jpg")
                        tags.append(idx)
                        tagname.append(tag)
                        count += 1 # otherwise overwrite the old bad file in the next loop
                    else:
                        fail += 1
    print("Done: %d failed" % fail)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        run(eval(sys.argv[1]))
