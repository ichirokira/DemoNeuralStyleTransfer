import streamlit as st
#from streamlit_webrtc import VideoTransformerFactory, webrtc_streamer
import torch
from utils import *
import tempfile
import cv2
import imutils

import tensorflow as tf

import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
st.header("Demo Style Transfer")
st.write("Written by TuyenNQ")



def app_image():
    model_dict = torch.load('./models/model_final.model')
    model_dict_clone = model_dict.copy()  # We can't mutate while iterating

    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model = Net(ngf=128)
    style_model.load_state_dict(model_dict, False)
    style_image_path = None


    style = st.sidebar.selectbox("Styles: ", ("None", "candy", "composition_vii", "escher_sphere", "feathers"
                                              ,"forest", "frida_kahlo", "house_water", "la_muse", "mosaic_ducks_massimo",
                                              "mosaic", "pencil", "picasso_selfport1907", "rain_princess", "Robert_Delaunay,_1906,_Portrait",
                                              "sea_rock", "sea", "seated-nude", "shipwreck", "starry_night", "stars2", "strp",
                                              "the_scream", "udnie", "wave", "woman-with-hat-matisse") )

    quotes = st.empty()
    start_button = st.empty()



    if style != "None":
        style_image_path = "./images/21styles/"+style
        if os.path.exists(style_image_path+".jpg"):
            style_image_path += ".jpg"
        else:
            style_image_path += ".jpeg"

        style_img = Image.open(style_image_path)
        st.sidebar.image(style_img)
    f = st.file_uploader("Your image", type=["png", "jpg", "jpeg"])
    if f is not None:

        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())

        img = Image.open(tfile.name)

        st.image(img)
        content_image = tensor_load_rgbimage(tfile.name, keep_asp=True).unsqueeze(0)

        if style_image_path is None:
            quotes.write("Please choose a Style")
        else:
            style = tensor_load_rgbimage(style_image_path).unsqueeze(0)
            style = preprocess_batch(style)

        start = start_button.button("Start")
        if start:
            quotes.write("Process your image......")

            style_v = Variable(style)
            content_image = Variable(preprocess_batch(content_image))

            style_model.setTarget(style_v)

            output = style_model(content_image)

            quotes.write("Here is your output:")
            img = output.data[0]
            (b, g, r) = torch.chunk(img, 3)
            img = torch.cat((r, g, b))
            img = img.clone().clamp(0, 255).numpy()
            img = img.transpose(1, 2, 0).astype('uint8')
            img = Image.fromarray(img)
            st.image(img, "Output")
        # tensor_save_bgrimage(output.data[0], 'output.jpg', False)

def app_makeup():
    quotes = st.empty()
    start_button = st.empty()
    style_image_path = None

    style = st.sidebar.selectbox("Styles: ", ("None", "vFG56", "vFG112", "vFG137", "vFG756", "vRX916", "XMY-014",
                                              "XMY-074", "XMY-136", "XMY-266"))
    if style != "None":
        style_image_path = "./makeup_images/makeup/"+style
        if os.path.exists(style_image_path+".png"):
            style_image_path += ".png"
        else:
            style_image_path += ".jpg"

        style_img = cv2.imread(style_image_path)
        show_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        show_img = Image.fromarray(show_img)
        st.sidebar.image(show_img)
    f = st.file_uploader("Your image", type=["png", "jpg", "jpeg"])
    if f is not None:

        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())

        img = cv2.imread(tfile.name)
        H,W = img.shape[:2]
        show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show_img = Image.fromarray(show_img)
        st.image(show_img)
        batch_size = 1
        img_size = 256
        no_makeup = cv2.resize(imread(tfile.name), (img_size, img_size))
        X_img = np.expand_dims(preprocess(no_makeup), 0)
        # makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
        result = np.ones((img_size, img_size, 3))
        # result[img_size: 2 * img_size, :img_size] = no_makeup / 255.

        tf.reset_default_graph()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph(os.path.join('./makeup_models', 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint('makeup_models'))

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')


        makeup = cv2.resize(imread(style_image_path), (img_size, img_size))
        Y_img = np.expand_dims(preprocess(makeup), 0)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = deprocess(Xs_)
        #print(Xs_.shape)
        result[:img_size, :img_size,:] = Xs_[0]
        result = cv2.resize(result, (W,H))
        #show_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        #show_img = Image.fromarray(show_img)
        #result = Image.fromarray(result)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(result.shape)
        #st.image(show_img, "Output")
        imsave('result.jpg', result)
        output = Image.open("result.jpg")
        st.image(output, "Output")
type = st.selectbox("Type: ", ("StyleTransfer", "Makeup"))

if type == "StyleTransfer":
    app_image()
else: 
    app_makeup()

