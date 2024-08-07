def recognizeHandGesture(landmarks):
  thumbState = 'UNKNOW'
  indexFingerState = 'UNKNOW'
  middleFingerState = 'UNKNOW'
  ringFingerState = 'UNKNOW'
  littleFingerState = 'UNKNOW'
  recognizedHandGesture = None

  pseudoFixKeyPoint = landmarks[2]['x']
  if (landmarks[3]['x'] < pseudoFixKeyPoint and landmarks[4]['x'] < landmarks[3]['x']):
    thumbState = 'CLOSE'
  elif (pseudoFixKeyPoint < landmarks[3]['x'] and landmarks[3]['x'] < landmarks[4]['x']):
    thumbState = 'OPEN'

  pseudoFixKeyPoint = landmarks[6]['y']
  if (landmarks[7]['y'] < pseudoFixKeyPoint and landmarks[8]['y'] < landmarks[7]['y']):
    indexFingerState = 'OPEN'
  elif (pseudoFixKeyPoint < landmarks[7]['y'] and landmarks[7]['y'] < landmarks[8]['y']):
    indexFingerState = 'CLOSE'

  pseudoFixKeyPoint = landmarks[10]['y']
  if (landmarks[11]['y'] < pseudoFixKeyPoint and landmarks[12]['y'] < landmarks[11]['y']):
    middleFingerState = 'OPEN'
  elif (pseudoFixKeyPoint < landmarks[11]['y'] and landmarks[11]['y'] < landmarks[12]['y']):
    middleFingerState = 'CLOSE'

  pseudoFixKeyPoint = landmarks[14]['y']
  if (landmarks[15]['y'] < pseudoFixKeyPoint and landmarks[16]['y'] < landmarks[15]['y']):
    ringFingerState = 'OPEN'
  elif (pseudoFixKeyPoint < landmarks[15]['y'] and landmarks[15]['y'] < landmarks[16]['y']):
    ringFingerState = 'CLOSE'

  pseudoFixKeyPoint = landmarks[18]['y']
  if (landmarks[19]['y'] < pseudoFixKeyPoint and landmarks[20]['y'] < landmarks[19]['y']):
    littleFingerState = 'OPEN'
  elif (pseudoFixKeyPoint < landmarks[19]['y'] and landmarks[19]['y'] < landmarks[20]['y']):
    littleFingerState = 'CLOSE'

  if (thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'OPEN' and ringFingerState == 'OPEN' and littleFingerState == 'OPEN'):
    recognizedHandGesture = 5
  elif (thumbState == 'CLOSE' and indexFingerState == 'OPEN' and middleFingerState == 'OPEN' and ringFingerState == 'OPEN' and littleFingerState == 'OPEN'):
    recognizedHandGesture = 4
  elif (thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'OPEN' and ringFingerState == 'CLOSE' and littleFingerState == 'CLOSE'):
    recognizedHandGesture = 3
  elif (thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'CLOSE' and ringFingerState == 'CLOSE' and littleFingerState == 'CLOSE'):
    recognizedHandGesture = 2
  elif (thumbState == 'OPEN' and indexFingerState == 'CLOSE' and middleFingerState == 'OPEN' and ringFingerState == 'CLOSE' and littleFingerState == 'CLOSE'):
    recognizedHandGesture = 10
  elif (thumbState == 'OPEN' and indexFingerState == 'OPEN' and middleFingerState == 'CLOSE' and ringFingerState == 'CLOSE' and littleFingerState == 'OPEN'):
    recognizedHandGesture = 7
  elif (thumbState == 'OPEN' and indexFingerState == 'CLOSE' and middleFingerState == 'CLOSE' and ringFingerState == 'CLOSE' and littleFingerState == 'OPEN'):
    recognizedHandGesture = 6
  else:
    recognizedHandGesture = 0
  return recognizedHandGesture

def getStructuredLandmarks(landmarks):
  structuredLandmarks = []
  for j in range(42):
    if( j %2 == 1):
      structuredLandmarks.append({ 'x': landmarks[j - 1], 'y': landmarks[j] })
  return structuredLandmarks
