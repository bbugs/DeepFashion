function DeFragTestEval(checkpoint)

% sus: this runs nicely
load(checkpoint, 'report');
params = report.params;
setupActFunc;
[oWe, ~, ~] = loadWordVectors(params);

test_split = getDatasetFold(params, 'test');

[e2r, e3r] = DeFragEval(test_split, oWe, params, report.theta, report.decodeInfo);
fprintf('%s test performance from %s:\n', params.dataset, checkpoint);
fprintf('e2 (image search)    : r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e2r<=1)*100, mean(e2r<=5)*100, mean(e2r<=10)*100, mean(e2r), floor(median(e2r)));
fprintf('e3 (image annotation): r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e3r<=1)*100, mean(e3r<=5)*100, mean(e3r<=10)*100, mean(e3r), floor(median(e3r)));
fprintf('-----\n');

% sus:
% e2 (image search)    : r@1: 29.4,	 r@5: 67.8,	 r@10: 82.6,	 rAVG: 7.4,	 rMED: 3
% e3 (image annotation): r@1: 35.0,	 r@5: 67.0,	 r@10: 86.0,	 rAVG: 10.6,	 rMED: 3


end