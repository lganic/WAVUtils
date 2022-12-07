import math
import numpy as np
from scipy.fft import fft, fftfreq
import pyaudio

_p = pyaudio.PyAudio()

def yieldAllInputDevices():
    deviceCount = _p.get_device_count()
    for deviceIndex in range(deviceCount):
        device = _p.get_device_info_by_index(index)
        if device.maxInputChannels > 0:
            yield device.name


class formatError(Exception):
    pass

class singleAudioStream:
    def __init__(self,bitrate,stream=None):
        self.stream=[]
        if stream!=None:
            self.stream=stream[:]
        self.bitrate=bitrate
    def normalize(self,value=1):
        v=max(max(self.stream),abs(min(self.stream)))
        v/=value
        for index, item in enumerate(self.stream):
            self.stream[index]=max(-1,min(1,item/v))
    def modulate(self,depth=100):
        originalStream=self.stream[:]
        for imd in range(2,depth+1):
            for index, item in enumerate(originalStream):
                self.stream[index]+=item**imd
    def length(self):
        return len(self.stream)/self.bitrate
    def save(self,name,detail=2):
        writeWav(name,self,detail)

class fullAudioStream:#multi channel audio
    def __init__(self,bitrate):
        self.streams=[]
        self.bitrate=bitrate
    def __add__(self,sas):
        if self.bitrate!=sas.bitrate:
            raise ValueError("Bitrates do not match")
        self.streams.append(sas)
    def __iadd__(self,sas):
        if self.bitrate!=sas.bitrate:
            raise ValueError("Bitrates do not match")
        self.streams.append(sas)
        return self
    def count(self):
        return len(self.streams)
    def __getitem__(self,index):
        return self.streams[index]
    def normalize(self,value=1):
        for stream in self.streams:
            stream.normalize(value)
    def modulate(self,depth=100):
        for stream in self.streams:
            stream.modulate(depth)
    def length(self):
        return len(self.streams[0].stream)/self.bitrate
    def merge(self):
        ns=[]
        streamCount=self.count()
        for streams in zip(*[singleStream.stream for singleStream in self.streams]):
            ns.append(sum(streams)/streamCount)
        return singleAudioStream(self.bitrate,ns)
    def save(self,name,detail=2):
        writeWav(name,self,detail)


def _bytesToNum(v):
    ret=0
    for i in v[::-1]:
        ret*=256
        ret+=i
    return ret

def _numToBytes(v,length=1):
    ret=b''
    while v>=256:
        ret+=bytes([v%256])
        v//=256
    ret+=bytes([v])
    return ret.ljust(length,b'\x00')

def readWav(fileName,secure=True):
    # i am well aware that my usage of bts=bts[x:] is wasteful, it just makes the code easier and is only used in the low cost area of processing the header
    if not fileName.endswith(".wav"):
        fileName+=".wav"
    bts=open(fileName,"rb").read()
    if secure and bts[:4]!=b'RIFF':
        raise formatError('First 4 bytes of file are not "RIFF"')
    bts=bts[4:]
    if secure and len(bts)-_bytesToNum(bts[:4])!=4:
        raise formatError('bytes 4-7 do not correctly describe file size')
    bts=bts[4:]
    if secure and bts[:8]!=b'WAVEfmt ':
        raise formatError('bytes 8-15 of file are not "WAVEfmt "')
    bts=bts[8:]
    wavSectionChunkSize=_bytesToNum(bts[:4])
    if secure and wavSectionChunkSize!=16:
        raise formatError('WAV section chunk size is not 16, while this isnt against WAV specification, I dont know how to load this specific header')
    bts=bts[4:]
    PCMCompressionType=_bytesToNum(bts[:2])
    if secure and PCMCompressionType!=1:
        raise formatError("Audio is compressed, and is not currently supported")
    bts=bts[2:]
    numberOfChannels=_bytesToNum(bts[:2])
    bts=bts[2:]
    sampFreq=_bytesToNum(bts[:4])
    bts=bts[4:]
    byteRate=_bytesToNum(bts[:4])
    bts=bts[4:]
    blockAllign=_bytesToNum(bts[:2])
    bts=bts[2:]
    bitsPerSample=_bytesToNum(bts[:2])
    if secure and byteRate!=(sampFreq*numberOfChannels*(bitsPerSample//8)):
        raise formatError("byte rate does not match with expected value, check byte rate, sample rate, number of channels, and bits per sample, ByteRate==SampleRate*NumChannels*(BitsPerSample/8)")
    if secure and blockAllign!=numberOfChannels*(bitsPerSample//8):
        raise formatError("block allign does not match with expected value, check block allign, number of channels, and bits per sample, BlockAlign=NumChannels*(BitsPerSample/8)")
    bts=bts[2:]
    if secure and bts[:4]!=b'data':
        raise formatError('bytes 36-39 if file are not "data"')
    bts=bts[4:]
    if secure and len(bts)-_bytesToNum(bts[:4])!=4:
        raise formatError('Subchunk size does not match data in subchunk header')
    bts=bts[4:]
    audioStreams=[]
    for _ in range(numberOfChannels):
        audioStreams.append(singleAudioStream(sampFreq))
    bytesPerStream=bitsPerSample//8
    conversionValue=2**(bitsPerSample-1)
    for chunkIndex in range(0,len(bts),blockAllign):
        chunk=bts[chunkIndex:chunkIndex+(bitsPerSample//8)]
        for streamIndex in range(0,numberOfChannels):
            if bytesPerStream!=2:
                audioStreams[streamIndex].stream.append((_bytesToNum(chunk[bytesPerStream*streamIndex:bytesPerStream*(streamIndex+1)])-conversionValue)/conversionValue)
            else:
                value=(_bytesToNum(chunk[bytesPerStream*streamIndex:bytesPerStream*(streamIndex+1)])-conversionValue)/conversionValue
                if value>=0:
                    value-=1
                else:
                    value+=1
                audioStreams[streamIndex].stream.append(value)
    output=fullAudioStream(sampFreq)
    for aSt in audioStreams:
        output+=aSt
    return output

def writeWav(fileName,sample,detail=2):
    if type(sample)==singleAudioStream:
        temp=fullAudioStream(sample.bitrate)
        temp+=sample
        sample=temp
    if not fileName.endswith(".wav"):
        fileName+=".wav"
    data=b""
    conversionValue=2**(8*detail-1)
    if detail==2:
        for stream in sample.streams:
            for index, item in enumerate(stream.stream):
                if item>=0:
                    item-=1
                else:
                    item+=1
                stream.stream[index]=item
    sampleArray=[[int(conversionValue*a+conversionValue) for a in stream.stream] for stream in sample.streams]
    for nSample in zip(*sampleArray):
        for sampleElement in nSample:
            data+=_numToBytes(sampleElement,detail)
    data=_numToBytes(len(data),4)+data#write data block length
    data=b'data'+data
    data=_numToBytes(detail*8,2)+data#write bits per sample
    data=_numToBytes(sample.count()*detail,2)+data#write block allign
    data=_numToBytes(sample.bitrate*sample.count()*detail,4)+data#write sample rate
    data=_numToBytes(sample.bitrate,4)+data#write bitrate
    data=_numToBytes(sample.count(),2)+data#write num channels
    data=_numToBytes(1,2)+data#write pcm
    data=_numToBytes(16,4)+data#write subchunk header size
    data=b'fmt '+data
    data=b'WAVE'+data
    data=_numToBytes(len(data),4)+data#write file size
    data=b'RIFF'+data
    file=open(fileName,"wb")
    file.write(data)
    file.close()

def fourierTransform(audioStream):
    if type(audioStream)==fullAudioStream:
        audioStream=audioStream.merge()
    L=len(audioStream.stream)
    spect=fft(audioStream.stream)[:L//2]
    freq=fftfreq(L,1/audioStream.bitrate)[:L//2]
    return (freq,abs(spect))



if __name__=="__main__":
    fname=input(">")
    if not fname.endswith(".wav"):
        fname+=".wav"
    stream=readWav(fname)
    stream.normalize(.9)
    freq, spect=fourierTransform(stream)
    f=open(fname.replace(".wav",".csv"),"w")
    for fr, s in zip(freq,spect):
        h=f.write(f"{fr},{s}\n")
    f.close()


if __name__=="__main__2":
    def map(targetMin,targetMax,dataMin,dataMax,value):#map value contained in range (dataMin,dataMax) to range of values (targetMin,targetMax)
        return ((value-dataMin)/(dataMax-dataMin))*(targetMax-targetMin)+targetMin
    bitrate=44100
    seconds=10
    frequency=1000
    array=[]
    import math
    for t in range(0,bitrate*seconds):
        array.append(math.sin((6.283*map(0,1000,0,bitrate*seconds,t)*t)/bitrate))
    t=singleAudioStream(bitrate,array)
    writeWav("test-detail-1",t,1)
    writeWav("test-detail-2",t,2)
    writeWav("test-detail-3",t,3)
    writeWav("test-detail-4",t,4)
