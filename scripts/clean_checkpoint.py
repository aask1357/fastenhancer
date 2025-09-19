from typing import Tuple, List
import os
import re
import argparse
from dataclasses import dataclass


DELETED_FILESIZE = 0
DELETE = False


@dataclass
class Node:
    name: str
    num_deleted: int
    subgraph: List['Node']


def print_filesize(filesize: int) -> None:
    print("Total deleted file size: ", end="")
    if filesize < 1024:
        print(f"{filesize} Bytes")
    elif filesize < 1024**2:
        print(f"{filesize/1024} KB")
    elif filesize < 1024**3:
        print(f"{filesize/1024**2} MB")
    elif filesize < 1024**4:
        print(f"{filesize/1024**3} GB")


def print_graph(graph: List[Node], prefix: str = "") -> None:
    for idx, node in enumerate(graph, start=1):
        if idx == len(graph):
            print(f"{prefix}└─", end="")
            subgraph_prefix = f"{prefix}   "
        else:
            print(f"{prefix}├─", end="")
            subgraph_prefix = f"{prefix}|  "
        if node.num_deleted > 0:
            print(f" ({node.num_deleted})", end="")
        print(f" {node.name}", end="\n")
        print_graph(node.subgraph, subgraph_prefix)


def make_graph(root: str, name: str) -> Tuple[Node, int]:
    global DELETED_FILESIZE
    global DELETE
    graph = []
    num_deleted = 0
    num_deleted_total = 0
    checkpoints = []
    for item in os.listdir(root):
        if os.path.isdir(os.path.join(root, item)):
            subgraph, num_deleted = make_graph(os.path.join(root, item), item)
            if num_deleted > 0:
                graph.append(subgraph)
                num_deleted_total += num_deleted
        else:
            if re.match('[0-9]{5,}.pth', item):
                checkpoints.append(int(item[:-4]))
    
    if len(checkpoints) <= 1:
        return Node(name, 0, graph), num_deleted_total
    
    checkpoints.sort()
    checkpoints_to_remove = checkpoints[:-1]
    for checkpoint in checkpoints_to_remove:
        DELETED_FILESIZE += os.path.getsize(os.path.join(root, f"{checkpoint:0>5d}.pth"))
        if DELETE:
            os.remove(os.path.join(root, f"{checkpoint:0>5d}.pth"))
    num_deleted_total += len(checkpoints_to_remove)
    return Node(name, len(checkpoints_to_remove), graph), num_deleted_total


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help="delete checkpoints")
    parser.add_argument('-n', '--names', nargs='+', type=str, help="directory names to clean")
    
    args = parser.parse_args()

    DELETE = args.delete
    
    graph = []
    for name in args.names:
        root = os.path.join("logs", name)
        node, num_deleted = make_graph(root, name)
        graph.append(node)
    print_graph(graph)
    print_filesize(DELETED_FILESIZE)
    