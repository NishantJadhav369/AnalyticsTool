css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: black;
}
.chat-message.bot {
    background-color: #2b313e
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://images.rawpixel.com/image_png_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTA4L3Jhd3BpeGVsb2ZmaWNlMTFfaWxsdXN0cmF0aW9uX29mX3RoZV9mdXR1cmVfaHVtYW5fcm9ib3Rfbm9fY3JvcF9kOGVhODEyMy0zNDQ2LTQwZmYtOThiMy1mMTRhNzQ3NTc2ODBfMS5wbmc.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="User.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
