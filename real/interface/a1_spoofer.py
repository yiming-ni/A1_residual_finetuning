import time
import numpy as np
import socket


class UDP:
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
        # self.s.sendto(message, ('172.17.0.2', 32768))

    def receive_wait(self):
        data, _ = self.receiver_socket.recvfrom(512)  # current buffer size is 1024 bytes
        return data  # return raw msg


class A1Spoofer():
    def __init__(self,
                 recv_IP="127.0.0.1", #"192.168.123.132",
                 recv_port=8001,
                 send_IP="127.0.0.1", #"192.168.123.12",
                 send_port=8000) -> None:
        self.udp = UDP(recv_IP, recv_port, send_IP, send_port)
        self._last_action = np.zeros((12,)) #np.concatenate([[1], np.zeros(3), np.zeros(3), np.zeros(12)], axis=-1)
        self._udp_init_done = False

    def send_obs(self, obs):
        if self._udp_init_done is False:
            obs = np.ones(30) * -10
        obs = np.round(obs, 5)  # 3
        obs = list(map(lambda x: str(x), list(obs)))
        obs += " "
        msg = " ".join(obs).encode('utf-8')
        self.udp.send_pack(msg)

    def receive_action(self):  # receive from gazebo and process it to array
        while 1:
            try:
                receive = self.udp.receive_wait()
                self._udp_init_done = True
                data = receive.split()
                data = list(map(lambda x: float(x), data))
                print("Received data", data)
                # print("thihg: ", data[8])
                self._last_action = data
                return data
            except:
                print("[ERROR]: Received No Data.")
                return self._last_action

            # rpy drpy mpos 0/1


if __name__ == '__main__':
    a1_spoofer = A1Spoofer()
    obs_spoofer = np.zeros((30,))
    while True:
        a1_spoofer.send_obs(obs_spoofer)
        a1_spoofer.receive_action()
        time.sleep(0.00005)
