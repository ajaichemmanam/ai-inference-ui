# python3.7 -m  streamlit run main.py
import os
import numpy as np
from PIL import Image
import streamlit as st

import numpy as np
import streamlit as st
import av
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

st.set_page_config(
    page_title="Inference Engine",
    layout="wide",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# WebRTC Config
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)

# VideoProcessor


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model) -> None:
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        img, labels = self.model.infer(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Load Models


# @st.cache
def load_poseestimation_model():
    print("Loading Tensorflow Pose Estimation model...")
    from models.pose_model import PoseEstimationModel
    pose_model = PoseEstimationModel()
    return pose_model


def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.title("AI Inference Engine")
    app_mode = st.sidebar.selectbox(
        "Choose a model", ["None",
                           "OpenPose-Openvino"])
    media_mode = st.sidebar.selectbox(
        "Media Source", ["None",
                         "Image",
                         "Webcam",
                         "Video"])

    if app_mode == "OpenPose-Openvino":
        st.title("Pose Detection")
        model = load_poseestimation_model()
        video_processor = VideoProcessor(model)

        if media_mode == "Image":
            img_file_buffer = st.file_uploader(
                "Upload an image", type=["png", "jpg", "jpeg"])
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer, mode='r')
                image = image.convert('RGB').resize((416, 416))
                original_image = np.copy(image)
                annotated_image, labels = model.infer(np.array(image))
                st.image([original_image, annotated_image], caption=[
                         "Original Image", "Inferred Image"], width=450)
                st.markdown("Objects detected")
                object_count = {}
                for i in labels:
                    if i not in object_count:
                        object_count[i] = 0
                    object_count[i] = object_count[i] + 1
                for object, count in object_count.items():
                    st.markdown("{} ----> {}".format(object, count))

        if media_mode == "Webcam":

            in_record_to = "./in.mp4"
            in_recorder_factory = None
            if st.checkbox("Record a video stream coming into the server"):

                def in_recorder_factory():
                    return MediaRecorder(str(in_record_to))

            out_record_to = "./out.mp4"
            out_recorder_factory = None
            if st.checkbox("Record a video stream going out from the server"):

                def out_recorder_factory():
                    return MediaRecorder(str(out_record_to))

            webrtc_ctx = webrtc_streamer(key="openpose filter", video_processor_factory=video_processor,
                                         mode=WebRtcMode.SENDRECV,
                                         client_settings=WEBRTC_CLIENT_SETTINGS,
                                         in_recorder_factory=in_recorder_factory,
                                         out_recorder_factory=out_recorder_factory,
                                         async_transform=True)
        elif media_mode == "Video":
            url = st.text_input('RTSP URL Link')
            st.text('OR')
            video_file = st.file_uploader('Select Video File', type=['mp4'])
            if video_file is not None:
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                media_file_info = {
                    # "path_or_url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
                    "path_or_url": tfile.name,
                    "type": "video",
                }
            elif url is not None:
                media_file_info = {
                    "path_or_url": url,
                    "type": "video",
                }

            if video_file is not None or url is not None:
                def create_player():
                    return MediaPlayer(str(media_file_info["path_or_url"]))

                WEBRTC_CLIENT_SETTINGS.update(
                    {
                        "media_stream_constraints": {
                            "video": media_file_info["type"] == "video",
                            "audio": media_file_info["type"] == "audio",
                        }
                    }
                )

                webrtc_ctx = webrtc_streamer(
                    key=f"media-streaming",
                    mode=WebRtcMode.RECVONLY,
                    client_settings=WEBRTC_CLIENT_SETTINGS,
                    player_factory=create_player,
                    video_processor_factory=video_processor,
                )


if __name__ == "__main__":
    main()
