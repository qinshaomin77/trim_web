#!/usr/bin/env python3
"""Merge script - restore fcd.xml"""
import os, glob

def merge_chunks():
    chunks = sorted(glob.glob('fcd.xml.part*'))
    if not chunks:
        print("Error: No chunk files found")
        return

    print(f"Found {len(chunks)} chunk files")
    with open('fcd.xml', 'wb') as outfile:
        for chunk in chunks:
            print(f'Merging: {chunk}')
            with open(chunk, 'rb') as infile:
                outfile.write(infile.read())

    print(f'\n✓ Restored: fcd.xml ({os.path.getsize("fcd.xml")/1024/1024:.2f} MB)')

if __name__ == '__main__':
    merge_chunks()
