#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

''' socket 是一种用于网络通信的编程接口，它是一种通信机制，用于在不同的计算机之间传输数据
    套接字提供了一种标准化的方法对不同计算机上的进程进行通信'''
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    '''将侦听器套接字绑定到指定的主机和端口，服务器将在这里侦听传入的连接。'''
    listener.bind((host, port))
    '''开始侦听传入的连接'''
    listener.listen()
    '''将侦听器套接字的超时设置为0秒，将超时设置为0意味着服务器的 accept() 方法将是非阻塞的，
    如果没有传入的连接，它将立即返回。'''
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        '''使用 listener 套接字的 accept() 方法接受来自客户端的新连接.如果有传入的连接， 
        accept() 方法返回一个新的套接字对象，该对象表示连接和客户端的地址.这些值分别赋给 conn 和 addr 变量.'''
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        '''连接套接字( conn )的超时设置为 None ，有效地禁用了超时'''
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    global conn
    '''从连接套接字  conn  中读取前 4 个字节的数据。'''
    messageLength = conn.recv(4)
    '''使用小端字节序将接收到的 4 个字节转换为一个整数值，这个整数值表示传入消息的长度。'''
    messageLength = int.from_bytes(messageLength, 'little')
    '''根据之前接收到的消息长度，从连接套接字  conn  中读取消息的剩余字节。'''
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    '''负责解析从客户端接收到的消息，并提取出相机参数和其他相关信息'''
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None