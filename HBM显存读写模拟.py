import torch
import math

def unpackbits(x):
    return torch.stack([x.byte() >> i & 1 for i in range(8)], dim=-1).bool()

def packbits(x):
    return (x.int() << torch.arange(8, device=x.device)).sum(-1).byte()

class DataMat:
    def __init__(self, size=512*512):
        self.size = size
        self.data = None

    def initialize(self):
        if self.data is None:
            self.data = torch.zeros(self.size, dtype=torch.uint8, device='cuda')

class Subarray:
    def __init__(self, num_mats=32):
        self.mats = [DataMat() for _ in range(num_mats)]
        self.row_buffer = None

    def initialize_row_buffer(self):
        if self.row_buffer is None:
            self.row_buffer = torch.zeros(2048, dtype=torch.uint8, device='cuda')  # 2KB row buffer

class Bank:
    def __init__(self, num_subarrays=32):
        self.subarrays = [Subarray() for _ in range(num_subarrays)]

class PseudoChannel:
    def __init__(self, num_banks=8):
        self.banks = [Bank() for _ in range(num_banks)]

class Channel:
    def __init__(self, size_mb=512):
        self.pseudo_channels = [PseudoChannel(), PseudoChannel()]

class HBM2Stack:
    def __init__(self, num_channels=8):
        self.channels = [Channel() for _ in range(num_channels)]
        self.original_size = 0

    def write(self, channel, pseudo_channel, bank, subarray, mat, address, data):
        self.original_size = data.size(0)  # Store the original size
        # Pad data to multiple of 32 bytes
        pad_size = (32 - data.size(-1) % 32) % 32
        padded_data = torch.nn.functional.pad(data, (0, pad_size))
        
        # Reshape to (A, 32)
        reshaped_data = padded_data.view(-1, 32)
        
        ecc_encoded_data = self.ecc_encode(reshaped_data)  # ECC encoding interface
        interleaved_data = self.interleave_data(ecc_encoded_data)
        
        target_mat = self.channels[channel].pseudo_channels[pseudo_channel].banks[bank].subarrays[subarray].mats[mat]
        if target_mat.data is None:
            target_mat.initialize()
        
        # Write all groups
        target_mat.data[address:address+interleaved_data.numel()] = interleaved_data.view(-1)

    def read(self, channel, pseudo_channel, bank, subarray, mat, address, num_groups):
        target_mat = self.channels[channel].pseudo_channels[pseudo_channel].banks[bank].subarrays[subarray].mats[mat]
        if target_mat.data is None:
            target_mat.initialize()
        
        interleaved_data = target_mat.data[address:address+num_groups*36].view(num_groups, 36)
        ecc_encoded_data = self.deinterleave_data(interleaved_data)
        return self.ecc_decode(ecc_encoded_data)  # ECC decoding interface

    def interleave_data(self, data):
        # Ensure data is 2D
        if data.dim() == 1:
            data = data.unsqueeze(0)
        # Interleave each group independently
        return torch.stack([self._interleave_group(group) for group in data])

    def _interleave_group(self, group):
        bits = unpackbits(group).view(-1)
        interleaved_bits = torch.zeros_like(bits)
        for i in range(288):
            new_index = (i * 73) % 288
            interleaved_bits[new_index] = bits[i]
        return packbits(interleaved_bits.view(-1, 8))

    def deinterleave_data(self, data):
        # Ensure data is 2D
        if data.dim() == 1:
            data = data.unsqueeze(0)
        # Deinterleave each group independently
        return torch.stack([self._deinterleave_group(group) for group in data])

    def _deinterleave_group(self, group):
        bits = unpackbits(group).view(-1)
        deinterleaved_bits = torch.zeros_like(bits)
        for i in range(288):
            original_index = (i * 73) % 288
            deinterleaved_bits[i] = bits[original_index]
        return packbits(deinterleaved_bits.view(-1, 8))

    def ecc_encode(self, data):
        # Simple ECC encoding: just add 4 bytes of zeros as a placeholder for each group
        ecc = torch.zeros(data.size(0), 4, dtype=torch.uint8, device='cuda')
        return torch.cat([data, ecc], dim=1)

    def ecc_decode(self, data):
        # Remove the last 4 bytes from each group
        decoded_data = data[:, :32].reshape(-1)
        # Remove any padding
        return decoded_data[:self.original_size]

def test_hbm2_structure():
    stack = HBM2Stack()

    # Test write and read with different input sizes
    test_sizes = [32, 64, 100]  # Test with different input sizes
    
    for size in test_sizes:
        print(f"\nTesting with input size: {size}")
        test_data = torch.arange(size, dtype=torch.uint8, device='cuda')
        print(f"Original test data: {test_data}")

        print("Writing data...")
        stack.write(0, 0, 0, 0, 0, 0, test_data)

        num_groups = math.ceil(size / 32)
        print("Reading data...")
        read_data = stack.read(0, 0, 0, 0, 0, 0, num_groups)

        print(f"Final read data: {read_data}")
        
        if torch.all(test_data.eq(read_data)):
            print("Test passed: Read data matches written data")
        else:
            print("Test failed: Read data doesn't match written data")
            print("Differences:")
            for i in range(size):
                if test_data[i] != read_data[i]:
                    print(f"Index {i}: Original {test_data[i]}, Read {read_data[i]}")

# Run the test
if torch.cuda.is_available():
    test_hbm2_structure()
else:
    print("CUDA is not available. Please run this on a CUDA-enabled machine.")
