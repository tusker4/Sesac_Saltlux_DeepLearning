from flask import Flask, render_template, request
# 1. 모듈 가져오기
from flask_socketio import SocketIO, emit
# 실시간 웹소켓기반 통신 -> 보안키 필요

app = Flask(__name__)
# 2. 보안키 설정
app.config['SECRET_KEY'] = 'ghkdrlfdl' # 임의의 값(해시값 범위내)
# 3. app을 소켓IO로 랲핑
socketio = SocketIO(app)



@app.route('/')
def home():
    return render_template('index.html')

# 챗봇 메인 페이지
@app.route('/nlp/chatbot')
def chatbot():
    return render_template('pages/chatbot.html')

# 4. 소켓 통신을 한 내용 구현
# 클라이언트가 보낸 내용을 서버가 받는다.
@socketio.on('cTos_send_msg')
def cTos_send_msg(data):
    print(type(data), data)




if __name__ == '__main__':
    # 5. 서버 가동을 소켓io 를 이용하여 구동교체
    # app.run(debug=True)
    socketio.run(app, debug=True)