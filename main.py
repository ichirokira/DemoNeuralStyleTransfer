import streamlit as st
from streamlit_webrtc import VideoTransformerFactory, webrtc_streamer
import torch
from utils import *
import tempfile
import cv2
import imutils

st.header("Demo Style Transfer")
st.write("Written by TuyenNQ")

model_dict = torch.load('./models/21styles.model')
model_dict_clone = model_dict.copy()  # We can't mutate while iterating

for key, value in model_dict_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del model_dict[key]
style_model = Net(ngf=128)
style_model.load_state_dict(model_dict, False)
def app_image():
    style_image_path = None

    style = st.sidebar.selectbox("Styles: ", ("None", "candy", "composition_vii", "escher_sphere", "feathers"
                                              ,"forest", "frida_kahlo", "house_water", "la_muse", "mosaic_ducks_massimo",
                                              "mosaic", "pencil", "picasso_selfport1907", "rain_princess", "Robert_Delaunay,_1906,_Portrait",
                                              "sea_rock", "sea", "seated-nude", "shipwreck", "starry_night", "stars2", "strp",
                                              "the_scream", "udnie", "wave", "woman-with-hat-matisse") )

    quotes = st.empty()
    start_button = st.empty()



    if style != "None":
        style_image_path = "./21styles/"+style
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


def app_video():
    f = st.file_uploader("Video", ["mp4", "avi"])
    style_loader = StyleLoader("./images/9styles", 512, False)
    style_model.eval()
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())
        vs = cv2.VideoCapture(tfile.name)
        idx = 0
        while True:
            # read frame
            idx += 1
            ret_val, img = vs.read()
            cimg = img.copy()
            img = np.array(img).transpose(2, 0, 1)
            # changing style
            if idx % 20 == 1:
                style_v = style_loader.get(int(idx / 20))
                style_v = Variable(style_v.data)
                style_model.setTarget(style_v)

            img = torch.from_numpy(img).unsqueeze(0).float()


            img = Variable(img)
            img = style_model(img)


                # simg = style_v.data().numpy()
            simg = style_v.data.numpy().reshape((3, 512, 512))
            img = img.clamp(0, 255).data[0].numpy()
            img = img.transpose(1, 2, 0).astype('uint8')
            simg = simg.transpose(1, 2, 0).astype('uint8')

            # display

            img = np.concatenate((img, simg), axis=1)
            st.image(img)
            # cv2.imwrite('stylized/%i.jpg'%idx,img)
            key = cv2.waitKey(1)

            if key == "q" or 0xFF:
                break
        vs.release()

        cv2.destroyAllWindows()

type = st.selectbox("Type: ", ("image", "video"))

if type == "image":
    app_image()
else: 
    app_video()