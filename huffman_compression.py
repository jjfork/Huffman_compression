import heapq
from typing import List

from tabulate import tabulate
import matplotlib.pyplot as plt


class Node:
    def __init__(self, symbol=None, count=None, left=None, right=None):
        self.symbol = symbol
        self.count = count
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.count < other.count


class HuffmanCompression:
    def __init__(self):
        self.tree = None
        self.codes = {}

    def compress(self, text: str) -> List[int]:
        freq_map = self._count_frequencies(text)
        self.tree = self._build_tree(freq_map)
        self._generate_code(self.tree, '')
        binary_string = self.binary_string(text)
        return binary_string

    def decompress(self, binary_data: List[int]) -> str:
        text = ''
        curr_node = self.tree
        for bit in binary_data:
            if bit == 0:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
            if self._is_leaf(curr_node):
                text += curr_node.symbol
                curr_node = self.tree
        return text

    def _count_frequencies(self, text: str) -> dict:
        freq_map = {}
        for symbol in text:
            if symbol not in freq_map:
                freq_map[symbol] = 0
            freq_map[symbol] += 1
        return freq_map

    def _build_tree(self, freq_map: dict) -> Node:
        priority_queue = []
        for symbol, count in freq_map.items():
            node = Node(symbol, count)
            heapq.heappush(priority_queue, node)

        while len(priority_queue) > 1:
            left_node = heapq.heappop(priority_queue)
            right_node = heapq.heappop(priority_queue)
            new_node = Node(left=left_node, right=right_node, count=left_node.count + right_node.count)
            heapq.heappush(priority_queue, new_node)

        return priority_queue[0]

    def _generate_code(self, node: Node, code: str):
        if self._is_leaf(node):
            self.codes[node.symbol] = code
        else:
            self._generate_code(node.left, code + '0')
            self._generate_code(node.right, code + '1')

    def _is_leaf(self, node: Node) -> bool:
        return node.left is None and node.right is None

    def binary_string(self, text: str):
        binary_string = []
        for symbol in text:
            code = self.codes[symbol]
            binary_string.extend([int(bit) for bit in code])
        return binary_string



# initialize the compression object
hc = HuffmanCompression()

# compress the text
result = hc.compress("Lorem ipsum dolor sit amet.")

# display the results as a table
table = [
    [symbol, count, ''.join(str(bit) for bit in code)]
    for symbol, count, code in result
]
print(tabulate(table, headers=["Symbol", "Count", "Huffman Code"]))

# display the compressed binary code
binary_code = hc.binary_string(result)
print("Compressed Binary Code:", binary_code)

# decode the compressed binary code and display the original text
decoded_text = hc.decompress(binary_code)
print("Decoded Text:", decoded_text)

# calculate the number of bits used before and after compression
original_size = len("Lorem ipsum dolor sit amet.") * 8
compressed_size = len(binary_code)
print("Original Size (in bits):", original_size)
print("Compressed Size (in bits):", compressed_size)

# calculate the compression ratio
compression_ratio = (1 - (compressed_size / original_size)) * 100
print("Compression Ratio (in %):", compression_ratio)

# plot the compression rate as a function of text length
text_lengths = [1, 3, 10, 25, 50]
bit_counts_before = []
bit_counts_after = []
compression_ratios = []
compression_times = []

for length in text_lengths:
    # generate sample text of given length
    text = "a" * length

    # compress the text and calculate metrics
    result = hc.compress(text)
    binary_code = hc.binary_string(result)
    decoded_text = hc.decompress(binary_code)
    original_size = length * 8
    compressed_size = len(binary_code)
    compression_ratio = (1 - (compressed_size / original_size)) * 100

    # append results to lists
    bit_counts_before.append(original_size)
    bit_counts_after.append(compressed_size)
    compression_ratios.append(compression_ratio)

# plot the bit count before and after compression
plt.plot(text_lengths, bit_counts_before, label="Before Compression")
plt.plot(text_lengths, bit_counts_after, label="After Compression")
plt.xlabel("Text Length (in characters)")
plt.ylabel("Bit Count (in bits)")
plt.title("Bit Count before and after Compression")
plt.legend()
plt.show()

# plot the compression ratio as a function of text length
plt.plot(text_lengths, compression_ratios)
plt.xlabel("Text Length (in characters)")
plt.ylabel("Compression Ratio (in %)")
plt.title("Compression Ratio as a function of Text Length")
plt.show()
