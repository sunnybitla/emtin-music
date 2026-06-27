import google.protobuf.message_factory
if not hasattr(google.protobuf.message_factory.MessageFactory, 'GetPrototype'):
	def GetPrototype(self, descriptor):
		return self.GetMessageClass(descriptor)
	google.protobuf.message_factory.MessageFactory.GetPrototype = GetPrototype

try:
	import google._upb._message as _upb
	if not hasattr(_upb.FieldDescriptor, 'label'):
		_upb.FieldDescriptor.label = property(lambda self: 3 if self.is_repeated else (2 if self.is_required else 1))
except Exception:
	pass

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

@st.cache_resource
def load_emotion_model():
	return load_model("model_fixed.keras")

@st.cache_resource
def load_holistic_model():
	return mp.solutions.holistic.Holistic()

model  = load_emotion_model()
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = load_holistic_model()              
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if emotion:
	st.info(f"Captured Emotion: **{emotion.upper()}**")

class EmotionProcessor:
	def __init__(self):
		self.frame_count = 0
		self.last_pred = ""
		self.last_res = None

	def recv(self, frame):
		self.frame_count += 1
		frm = frame.to_ndarray(format="bgr24")
		frm = cv2.flip(frm, 1)

		# Only run MediaPipe and prediction every 10 frames (~3 times per second)
		# to optimize CPU and prevent lag
		if self.frame_count % 10 == 0 or self.last_res is None:
			res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
			self.last_res = res
			
			if res.face_landmarks:
				lst = []
				for i in res.face_landmarks.landmark:
					lst.append(i.x - res.face_landmarks.landmark[1].x)
					lst.append(i.y - res.face_landmarks.landmark[1].y)

				if res.left_hand_landmarks:
					for i in res.left_hand_landmarks.landmark:
						lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
						lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
				else:
					for i in range(42):
						lst.append(0.0)

				if res.right_hand_landmarks:
					for i in res.right_hand_landmarks.landmark:
						lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
						lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
				else:
					for i in range(42):
						lst.append(0.0)

				lst = np.array(lst).reshape(1,-1)
				pred = label[np.argmax(model.predict(lst))]
				self.last_pred = pred
				print("Predicted emotion:", pred)
				np.save("emotion.npy", np.array([pred]))
		else:
			res = self.last_res

		pred = self.last_pred
		if pred:
			cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		if res:
			if res.face_landmarks:
				drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
										landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
										connection_drawing_spec=drawing.DrawingSpec(thickness=1))
			if res.left_hand_landmarks:
				drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
			if res.right_hand_landmarks:
				drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer:
	webrtc_streamer(
		key="key",
		desired_playing_state=True,
		video_processor_factory=EmotionProcessor,
		media_stream_constraints={"video": True, "audio": False},
		rtc_configuration=RTC_CONFIGURATION
	)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
	else:
		url = f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
		st.success(f"Recommending {emotion} songs by {singer} in {lang}!")
		
		# Show a beautiful direct link
		st.markdown(f"""
			<div style="margin: 15px 0;">
				<a href="{url}" target="_blank" style="text-decoration: none;">
					<div style="background-color: #FF0000; color: white; padding: 12px 24px; text-align: center; border-radius: 8px; font-weight: bold; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;">
						🎵 Click here to open songs on YouTube 🎵
					</div>
				</a>
			</div>
		""", unsafe_allow_html=True)
		
		# Auto-open in new tab using Javascript
		js = f"window.open('{url}', '_blank');"
		st.components.v1.html(f"<script>{js}</script>", height=0)
		
		np.save("emotion.npy", np.array([""]))
