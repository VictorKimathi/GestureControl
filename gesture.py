import cv2
import mediapipe as mp
from func import recognizeHandGesture, getStructuredLandmarks
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up Firefox options
firefox_options = FirefoxOptions()
firefox_options.add_argument('--no-sandbox')
firefox_options.add_argument('--disable-dev-shm-usage')

# Set up the FirefoxDriver service
service = FirefoxService(GeckoDriverManager().install())

try:
    # Initialize the WebDriver with options and service
    driver = webdriver.Firefox(service=service, options=firefox_options)

    def gest():
        try:
            # Open YouTube video and start playback
            driver.get('https://www.youtube.com/watch?v=3WCIyNOrzwM')
            driver.execute_script('document.getElementsByTagName("video")[0].play()')
            
            # Set up MediaPipe Hands
            hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                l = []
                success, image = cap.read()
                if not success:
                    print("Failed to grab frame.")
                    break
                
                # Process the image with MediaPipe
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                        for lp in hand_landmarks.landmark:
                            l.append(lp.x)
                            l.append(lp.y)
                
                cv2.imshow('MediaPipe Hands', image)
                
                # Recognize hand gesture and control video
                if l:  # Check if landmarks are available
                    try:
                        recognizedHandGesture = recognizeHandGesture(getStructuredLandmarks(l))
                        print(f"Recognized Gesture: {recognizedHandGesture}")

                        if recognizedHandGesture == 5:
                            driver.execute_script('document.getElementsByTagName("video")[0].pause()')
                        elif recognizedHandGesture == 4:
                            driver.execute_script('document.getElementsByTagName("video")[0].play()')
                        elif recognizedHandGesture == 3:
                            driver.execute_script('document.getElementsByTagName("video")[0].playbackRate = 2')
                        elif recognizedHandGesture == 2:
                            driver.execute_script('document.getElementsByTagName("video")[0].playbackRate = 1')
                        elif recognizedHandGesture == 6:
                            driver.execute_script('document.getElementsByTagName("video")[0].volume = 0')
                        elif recognizedHandGesture == 7:
                            driver.execute_script('document.getElementsByTagName("video")[0].volume = 1')
                    except Exception as e:
                        print(f"Error processing hand gesture: {e}")
                else:
                    print("No hand landmarks detected.")
                
                # Exit loop if 'ESC' key is pressed
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            # Cleanup
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            driver.quit()

    gest()
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
