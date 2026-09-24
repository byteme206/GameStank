# GameStank

## Overview
This project is an early attempt to solve for purchasing a GameScent hardware room scenting device that was bricked before delivery by the collapse of its backing business. 

Rather than waste the investment, this is my attempt to create a locally running AI service that analyzes game audio in 1-second chunks and labels sounds accordingly. If a matching label is found, it triggers the scenting device to fire a corresponding atomizer to dispense essential oil into the air.

## Status
Right now, this is an application skeleton. It has not been tested or even properly developed. I need to find someone with a working device willing to run Wireshark packet capture to suss out whether the device uses raw UDP datagrams or JSON payloads, and if the latter, what the shape of the requests needs to look like.

The tensorflow and YAMnet logic should work and apply to the default scent bottle slot mappings.