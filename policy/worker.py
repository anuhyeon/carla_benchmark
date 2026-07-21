#!/usr/bin/env python3

import argparse

import zmq

from protocol import receive, send


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--policy", choices=["citywalker", "genie_samtp"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["rgb", "rgbd"], default="rgb")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.policy == "citywalker":
        from citywalker import CityWalkerPolicy

        policy = CityWalkerPolicy(args.config, args.checkpoint, args.device)
    else:
        from genie_samtp import GeniePolicy

        policy = GeniePolicy(args.config, args.checkpoint, args.mode, args.device)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.endpoint)

    while True:
        request = receive(socket) #클라이언트로 부터 리퀘스트 받음 예{'type': 'describe'} 아마 클라이언트는 receive()에서 계속 대기중임 
        if request["type"] == "describe":
            response = policy.describe()
        elif request["type"] == "reset":
            response = policy.reset(request["episode"])
        elif request["type"] == "step":
            response = policy.step(request["observation"], request["subgoal"])
        else:
            send(socket, {"status": "closed"})
            break
        send(socket, response)

    socket.close()
    context.term()


if __name__ == "__main__":
    main()
