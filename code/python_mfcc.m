function [ CC, FBE, frames ] = runmfcc( speech )

opt.window   = [0 1];
opt.fs       = 16000;
opt.Tw       = 25;
opt.Ts       = 10;            % analysis frame shift (ms)
opt.alpha    = 0.97;          % preemphasis coefficient
opt.R        = [ 300 3700 ];  % frequency range to consider
opt.M        = 40;            % number of filterbank channels
opt.C        = 13;            % number of cepstral coefficients
opt.L        = 22;            % cepstral sine lifter parameter


N=opt.C;
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));


[CC, FBE, frames] = mfcc( speech, opt.fs, opt.Tw, opt.Ts, opt.alpha, hamming, opt.R, opt.M, N, opt.L );

end