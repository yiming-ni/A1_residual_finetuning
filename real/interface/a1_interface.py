import time
import numpy as np
import socket


class UDP_for_net:
    def __init__(self,
                 recv_IP='127.0.0.1',
                 recv_port=8000,
                 send_IP='127.0.0.1',
                 send_port=8001):
        self.sender_IP = send_IP
        self.UDP_PORT_SENDING = send_port
        # self.sending_freq = 2000.0
        self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sender_socket.settimeout(0.1)
        self.sender_addr = (self.sender_IP, self.UDP_PORT_SENDING)
        print("UDP Sender is initialized:", send_IP)

        self.receiver_IP = recv_IP
        self.UDP_PORT_RECVING = recv_port
        self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_socket.settimeout(0.1)
        self.receiver_socket.bind((self.receiver_IP, self.UDP_PORT_RECVING))
        print("UDP Receiver is initialized:", recv_IP)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)

    def send_pack(self, message):
        # print("Sending")
        self.sender_socket.sendto(message, self.sender_addr)
        self.s.sendto(message, ('172.17.0.2', 32768))

    def receive_wait(self):
        data, _ = self.receiver_socket.recvfrom(512)  # current buffer size is 1024 bytes
        return data  # return raw msg


class A1Interface():
    def __init__(self,
                 recv_IP="192.168.123.132",
                 recv_port=32770,
                 send_IP="192.168.123.12",
                 send_port=32769) -> None:
        self.udp = UDP_for_net(recv_IP, recv_port, send_IP, send_port)
        self._last_obs = np.concatenate([[1], np.zeros(3), np.zeros(3), np.zeros(12)], axis=-1)
        self._udp_init_done = False

    def send_command(self, action):
        if self._udp_init_done is False:
            action = np.ones(12) * -10
        # print("action_raw",action)
        # print("already connected? ",self._udp_init_done)
        action = np.round(action, 5)  # 3
        action = list(map(lambda x: str(x), list(action)))
        action += " "
        msg = " ".join(action).encode('utf-8')
        self.udp.send_pack(msg)

    def receive_observation(self):  # receive from gazebo and process it to array
        while 1:
            try:
                receive = self.udp.receive_wait()
                self._udp_init_done = True
                data = receive.split()
                data = list(map(lambda x: float(x), data))
                # print("Received data", data)
                # print("thihg: ", data[8])
                self._last_obs = data
                return data
            except:
                print("Returning last observation.")
                return self._last_obs

            # rpy drpy mpos 0/1
