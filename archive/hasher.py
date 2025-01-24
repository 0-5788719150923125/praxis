class FastCompressor:
    def __init__(self):
        self.input_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?:/"
        self.char_to_val = {c: i for i, c in enumerate(self.input_chars)}
        self.val_to_char = {i: c for i, c in enumerate(self.input_chars)}
        self.input_base = len(self.input_chars)  # 55
        self.chars_per_group = 2  # Reduced from 3 to 2 for safety
        self.max_value = 0x10FFFF - 0x1000  # Maximum safe Unicode value

    def encode(self, input_str):
        result = []

        # Process in groups of 2
        for i in range(0, len(input_str), self.chars_per_group):
            group = input_str[i : i + self.chars_per_group]

            # Convert group to single value
            value = 0
            for c in group:
                value = value * self.input_base + self.char_to_val[c]

            # Add padding information (using fewer bits since values are smaller)
            padding = self.chars_per_group - len(group)
            if padding:
                value |= padding << 16  # Reduced from 20 to 16 bits

            # Verify we're in safe range
            if value + 0x1000 >= 0x110000:
                raise ValueError(
                    f"Encoded value {value + 0x1000:X} exceeds Unicode maximum"
                )

            # Convert to Unicode char
            result.append(chr(value + 0x1000))

        return "".join(result)

    def decode(self, encoded_str):
        result = []

        for c in encoded_str:
            # Get value back
            value = ord(c) - 0x1000

            # Extract padding
            padding = value >> 16  # Reduced from 20 to 16 bits
            value &= (1 << 16) - 1  # Clear padding bits

            # Calculate how many chars were in this group
            chars_in_group = self.chars_per_group - (padding or 0)

            # Convert back to characters
            temp = []
            for _ in range(chars_in_group):
                value, remainder = divmod(value, self.input_base)
                temp.append(self.val_to_char[remainder])
            result.extend(reversed(temp))

        return "".join(result)


# Test implementation
def test_fast_compressor():
    compressor = FastCompressor()

    test_strings = [
        "ABC",
        "HelloWorld",
        "This/is:a?test/string:with?special/characters",
        "A" * 60,
        "Z" * 100,
    ]

    for original in test_strings:
        try:
            encoded = compressor.encode(original)
            decoded = compressor.decode(encoded)
            print(f"\nOriginal ({len(original)} chars): {original}")
            print(f"Encoded ({len(encoded)} chars): {encoded}")
            print(f"Decoded ({len(decoded)} chars): {decoded}")
            print(f"Correctly decoded: {original == decoded}")
            print(f"Compression ratio: {len(encoded) / len(original):.2f}")
        except Exception as e:
            print(f"Error with string {original}: {str(e)}")


test_fast_compressor()
