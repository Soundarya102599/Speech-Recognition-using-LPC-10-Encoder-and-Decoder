filename ="input_file.wav";
[a ,periods,power,voicingInd,unvoicingInd, windowSize ] = mainEncoder( filename );
%% save variables
delete 'pack.mat';
save('pack.mat','a','periods','power','voicingInd','unvoicingInd', 'windowSize');
load('pack.mat');


%% reconstruct
reconstSig = mainDecoder(a(2:end,:),periods, power, voicingInd, unvoicingInd, windowSize);
rate = 8000;
soundsc(reconstSig,rate);

%% save reconstructed sound
reconstSig = reconstSig./max(reconstSig).*0.8;
audiowrite('reconstructed.wav',reconstSig,rate);
subplot(2,1,2);
plot(reconstSig)
xlabel('Time');
ylabel('Signal Amplitude');
title('Synthesized Speech')


function [ a,periods,power,voicingInd,unvoicingInd, windowSize ] = mainEncoder( filename )
rate = 8000;
windowSize = 180;
speechThresh = 0.008;
%voicedThresh = 0.001;   %decreasing thresh gives more voiced frames.
voicingThresh = 80;%decreasing thresh gives more unvoiced frames.

%% load and preprocess audio data
signal = getAudio(filename,rate);
subplot(2,1,1);
plot(signal)
xlabel('Time');
ylabel('Signal Amplitude');
title('Original Speech')

signal = preEmphasis(signal);
frames = getSegment(signal,windowSize);
%signal = frames2signal(frames); %padding to signal.

%% split frames into non-speech, voiced speech and unvoiced speech
[speechFrames, speechInd] = speechDetector(frames,speechThresh);
%[voicingFrames_energy, voicingInd_energy] = speechDetector(speechFrames,voicedThresh);%voicingDetector(speechFrames,rate);
[~, voicingInd, ~] = voicingDetector( speechFrames, speechInd, rate, voicingThresh );
%voicingInd = voicingInd & voicingInd_energy;
%voicingFrames(:,voicingInd) = speechFrames(:,voicingInd);

unvoicingInd = xor(speechInd,voicingInd);
%unvoicingFrames = frames;
%unvoicingFrames(:,~unvoicingInd) = 0;

%% LP analysis
[a,~,~,~] = levinsonDurbin(speechFrames, 10);

%% prediction error
errorFrames =predictionErrorFilter(speechFrames,a(2:end,:));


%% period
periods = pitchPeriodEstimator(errorFrames,voicingInd, rate);

%% power computation
power = powerComputation(errorFrames,periods,unvoicingInd, voicingInd);

end

function reconstSig = mainDecoder(a,periods, power, voicingInd, unvoicingInd, windowSize)
    reconstSig = zeros(size(power,2)*windowSize,1);
    i = 1;
    while i<=length(reconstSig)
        frameId = ceil(i/windowSize);
        if voicingInd(frameId)
            T = periods(:,frameId);
            p = power(frameId);
         
             %% test begin
            tCenter = mod(i,windowSize)+T/2;
            if tCenter>=windowSize/2 && frameId+1<=size(voicingInd,2) && voicingInd(frameId+1)
                alpha = tCenter/windowSize-0.5;     %0~0.5
                T = round((0.5+alpha)*T+(0.5-alpha)*periods(:,frameId+1));
                p = (0.5+alpha)*p+(0.5-alpha)*power(frameId+1);
            end
            if tCenter<windowSize/2 && frameId-1>=1 && voicingInd(frameId-1)
                alpha = 0.5-tCenter/windowSize; %0~0.5
                T = round((0.5+alpha)*T+(0.5-alpha)*periods(frameId-1));
                p = (0.5+alpha)*p+(0.5-alpha)*power(:,frameId-1);
            end
          %% test end
          
            impulse = [0.5*sqrt(p);zeros(T-1,1)];
            B = 1;
            A = [1;-a(:,frameId)];
            pitch = filter(B,A,impulse);
            pitch = pitch - mean(pitch);
            reconstSig(i:min(i+T-1,length(reconstSig)),1) = pitch(1:min(i+T-1,length(reconstSig))-i+1);
            i = i+T;
        else if unvoicingInd(frameId)% || (voicingInd(frameId) && mod(i,windowSize)+T/2>windowSize)
            whiteNoise = randn(windowSize,1)*power(frameId);
            B = 1;
            A = [1;-a(:,frameId)];
            colorfulNoise = filter(B,A,whiteNoise);
            colorfulNoise = colorfulNoise-mean(colorfulNoise);
            reconstSig(i:min(i+windowSize-1,length(reconstSig)),1) = colorfulNoise(1:min(i+windowSize-1,length(reconstSig))-i+1);
            i = i+windowSize;
        else
            i = windowSize*frameId+1;
            end
        end
    end
    reconstSig = DeEmphasis(reconstSig,0.9375);
end
function [ y ] = DeEmphasis( x, alpha )
    if nargin<2
        alpha = 0.9375;
    end
    b = 1;
    a = [1,-alpha];
    y = filter(b,a,x);
end

function [ signal ] = frames2signal( frames)
    sz = size(frames);
    signal = reshape(frames,sz(1)*sz(2),1);
end

function [ s ] = getAudio( filename,rate )
    [s,fs] = audioread(filename);
    t = floor(1:(fs/rate):length(s));
    s = s(t);
    if(size(s,1)<3)
        s = s';
    end
end
function energy = getEnergy(frames)
    sz = size(frames);
    energy = sum(frames.^2)/sz(1);
end

function frames = getSegment( s, windowSize )
    % make sure s is a column vector
    sz = size(s);
    if(sz(1)<sz(2))
        s = s';
    end
    
    %padding and reshape
    l = ceil(length(s)/windowSize)*windowSize;
    s = padarray(s,l-length(s),'post');
    frames = reshape(s,windowSize,l/windowSize);
end

function [ sig ] = ind2signal( ind, windowSize )
    sig = reshape(repmat(ind,windowSize,1),windowSize*length(ind),1);
end

function [a,e,k,R] = levinsonDurbin( frames, order )
    sz = size(frames);
    R = getHalfAutocorrelation(frames, order);
    a = [ones(sz(2),1),zeros(sz(2),order)];
    e = [R(1,:)',zeros(sz(2),order)];
    k = zeros(sz(2),order+1);
    for i = 1:order
        k(:,i+1) = (R(i+1,:)'-diag(a(:,2:i)*flipud(R(2:i,:))))./e(:,i);
        a(:,2:i) = a(:,2:i)-bsxfun(@times,k(:,i+1),fliplr(a(:,2:i)));
        a(:,i+1) = k(:,i+1);
        e(:,i+1) = (1-k(:,i+1).^2).*e(:,i);
    end
    k = k';
    a = a';
    e = e';
end

function R = getHalfAutocorrelation(frames, order)
    R = zeros(order+1,size(frames,2));
    for f = 1:size(frames,2)
        ac = xcorr(frames(:,f));
        R(:,f) = ac(size(frames,1):size(frames,1)+order);
    end
end

function [lags, minMdf] = pitchPeriodEstimator( frames, voicingInd, rate)
    if nargin < 2
        voicingInd = ones(1,size(frames,2));
    end
    if nargin < 3
        rate = 8000;
    end
    windowSize = size(frames,1);
    frames = frames(:,voicingInd);
    voicingLags = zeros(1,size(frames,2));
    voicingMinMdf = Inf(1,size(frames,2));
    startLag = 20*rate/8000;
    for l = startLag:ceil(windowSize/2)
        mdf = mean(abs(frames(1+l:end,:)-frames(1:end-l,:)));
        %Should the above be mean or sum? The textbook example says sum,
        %but I think mean makes more sense.
        smallerMdfInd = mdf<voicingMinMdf;
        voicingMinMdf(smallerMdfInd) = mdf(smallerMdfInd);
        voicingLags(smallerMdfInd) = l;
    end
    lags = zeros(1,size(voicingInd,2));
    minMdf = Inf(1,size(voicingInd,2));
    lags(voicingInd) = voicingLags;
    minMdf(voicingInd) = voicingMinMdf;
end

function [ power ] = powerComputation(errorFrames, periods,unvoicingInd,voicingInd) 
    power = zeros(1,size(errorFrames,2));
    %% power of unvoiced frames
    uFrames = errorFrames(:,unvoicingInd);
    power(:,unvoicingInd) = mean(uFrames.^2);
    
    %% power of voicied frames
    windowSize = size(errorFrames,1);
    for col = 1:size(errorFrames,2)
        if voicingInd(col)~=0
            stop = windowSize*mod(periods(col),windowSize);
            errorFrames(stop+1:end,col) = 0;
        end
    end
    vFrames = errorFrames(:,voicingInd);
    power(:,voicingInd) = mean(vFrames.^2);
end

function [ y ] = preEmphasis( x, alpha )
    if nargin<2
        alpha = 0.9375;
    end
    b = [1,-alpha];
    a = 1;
    y = filter(b,a,x);
end

function errors = predictionErrorFilter( frames, a )
    errors = zeros(size(frames));
    for col = 1:size(frames,2)
        B = [1;-a(:,col)];
        A = 1;
        errors(:,col) = filter(B,A,frames(:,col));
    end
end

function [ speech, speechInd] = speechDetector( frames, thresholdRatio)
    energy = getEnergy(frames);
    if nargin <2
        thresholdRatio = 0.01;
    end;
    speechInd = energy>max(energy)*thresholdRatio;
    speech = frames;
    speech(:,~speechInd)=0;
end

function [ voicingFrames, voicingInd, zeroCrossCnt] = voicingDetector( frames, speechInd, rate, voicingThresh )
    %threshF = 3000;%Energy for voiced speech tends to concentrate below 3KHz.
    if nargin<3
        rate = 8000;
    end
    if nargin<4
        voicingThresh = 21;
    end
    thresh = voicingThresh*(size(frames,1)/180)/(rate/8000);
    sgn = frames>0;
    cross = xor(sgn(1:end-1,:),sgn(2:end,:));
    zeroCrossCnt = sum(cross);
    voicingInd = zeroCrossCnt<thresh;
    voicingInd = voicingInd & speechInd;
    voicingFrames = frames;
    voicingFrames(:,~voicingInd) = 0;
end