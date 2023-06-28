import streamlit as st
import cv2
import tempfile
from trackTruck import playVideo, trackTruck

def showFrame(frame):
    st_frame.image(frame[:, :, ::-1])
    return True
def processFrame(frame):
    return trackTruck(h_bg, frame)

if __name__ == "__main__":
    st.title("Track truck on traffic videos using OpenCV")

    source = st.selectbox('choose a traffic video file from', ('local', 'server'))

    video_file = None
    if source == 'local':
        uploaded_file = st.file_uploader("choose a traffic video file", type=['mp4'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                video_file = temp_file.name
    elif source == 'server':
        video_file = 'Traffic Videos/test.mp4'

    if video_file:
        st_frame = st.empty()
        h_bg = cv2.createBackgroundSubtractorKNN(history=200)
        playVideo(video_file, processFrame, showFrame)
        # h_video = cv2.VideoCapture(video_file)

        # st.write('codec', int(h_video.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode())
        #
        # h_video.release()
        # exit()


