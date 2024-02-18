# -*- encoding: utf-8 -*-
# 测试环境: Python epoch5.epoch1_3000 版本

import socket
import sys


host = "192.168.2.1"
port = 40923


def main():

    address = (host, int(port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting...")
    s.connect(address)
    print("Connected!")
    while True:
        msg = input(">>> please input SDK cmd: ")
        if msg.upper() == "Q":
            break
        msg += ";"
        s.send(msg.encode("utf-8"))

        try:
            buf = s.recv(1024)
            print(buf.decode("utf-8"))
        except socket.error as e:
            print("Error receiving :", e)
            sys.exit(1)
        if not len(buf):
            break
    s.shutdown(socket.SHUT_WR)
    s.close()


if __name__ == "__main__":
    main()
