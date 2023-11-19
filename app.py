import streamlit as st
import os
print(os.listdir())
from zero_shot_tracking.track import combine_all

st.write("Object Measurement")

out_zip_path = "out.zip"
out_display_video_path = "out_display.mp4"
input_video_path = st.text_input("enter the input video path")
if input_video_path:
  st.video(input_video_path)
  out_video_path = "out_measure.mp4"
  TEXT_PROMT = st.text_input("enter the object names to detect")
  BOX_THERSOLD = st.text_input("enter the box ther")

  if all([out_video_path,TEXT_PROMT,BOX_THERSOLD]):
    combine_all(input_video_path,TEXT_PROMT,float(BOX_THERSOLD),
          0.4,out_video_path,out_display_video_path)

  if os.path.isfile(out_display_video_path):st.video(out_display_video_path)
