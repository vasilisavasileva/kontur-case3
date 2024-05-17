import streamlit as st
import pandas as pd
import numpy as np

	#### IMPORT #####
import cv2
import time

import plotly.express as px


def main():
	st.title("Бля")
	
	### CONNECT TO CAMERA 
	# either like this using your own camera IP
	capture = cv2.VideoCapture('rtsp://192.168.1.64/1')

	### MAKE PLACE HOLDER FOR VIDEO FRAMES
	FRAME_WINDOW =st.image([])

	### GRAB NEW IMAGE
	x, frame = camera.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	time.sleep(0.025)
	

if __name__ == "__main__":
	main()
