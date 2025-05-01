'''thveissojamkddid'''

import smtplib
from email.mime.text import MIMEText
from email.header import Header

def send_email(sender = "2832941153@qq.com",
               receiver = "2832941153@qq.com",
               subject = "训练提示",
               body = '训练完成',
               password = "thveissojamkddid"):
    # 邮件内容设置
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = Header(subject, 'utf-8')

    try:
        # 连接到QQ邮箱的SMTP服务器
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        # 登录邮箱
        server.login(sender, password)
        # 发送邮件
        server.sendmail(sender, receiver, msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败，错误信息：{str(e)}")
    finally:
        # 关闭连接
        server.quit()

# 使用示例
if __name__ == "__main__":
    # 你的QQ邮箱地址
    sender_email = "2832941153@qq.com"
    # 接收者邮箱（这里是给自己发，所以和发送者相同）
    receiver_email = "2832941153@qq.com"
    # 邮件主题
    subject = "测试邮件"
    # 邮件正文
    body = "这是一封使用Python发送的测试邮件！"
    # 你的QQ邮箱SMTP授权码（不是邮箱密码）
    password = "thveissojamkddid"

    # send_email(sender_email, receiver_email, subject, body, password)
    send_email()