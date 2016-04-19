function examineReport(report_path)
% helper function that loads a report (which contains the model checkpoint)
% and other training statistics and prints it

if ~exist(report_path, 'file')
    % try looking in cv
    report_path = ['cv/' report_path];    
end
load(report_path); % loads report struct

report.params
iter = report.iter;
maxiters = report.maxiters;
current_epoch = iter/maxiters*report.params.maxepochs;
fprintf('loaded checkpoint form iter %d/%d, epoch %.2f\n', iter, maxiters, current_epoch);

% plot costs
figure(1)
subplot(221)
plot((1:iter)/maxiters*report.params.maxepochs, report.raw_costs(1:iter), 'b');
xlabel('epoch');
title('raw cost cost');

subplot(222)
plot((1:iter)/maxiters*report.params.maxepochs, report.reg_costs(1:iter), 'r');
title('regularization cost');

subplot(223)
xa = report.hist_iter/maxiters*report.params.maxepochs;
plot(xa, report.hist_e2t, 'b', xa, report.hist_e2v, 'r');
legend('train', 'val');
title('image search R@10 score');

subplot(224);
xa = report.hist_iter/maxiters*report.params.maxepochs;
plot(xa, report.hist_e3t, 'b', xa, report.hist_e3v, 'r');
legend('train', 'val');
title('image annotation R@10 score');

% plot final statistics
e2r = report.train_e2r;
e3r = report.train_e3r;
fprintf('training performance:\n');
fprintf('e2: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e2r<=1)*100, mean(e2r<=5)*100, mean(e2r<=10)*100, mean(e2r), floor(median(e2r)));
fprintf('e3: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e3r<=1)*100, mean(e3r<=5)*100, mean(e3r<=10)*100, mean(e3r), floor(median(e3r)));
fprintf('-----\n');

e2r = report.val_e2r;
e3r = report.val_e3r;
fprintf('validation performance:\n');
fprintf('e2: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e2r<=1)*100, mean(e2r<=5)*100, mean(e2r<=10)*100, mean(e2r), floor(median(e2r)));
fprintf('e3: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e3r<=1)*100, mean(e3r<=5)*100, mean(e3r<=10)*100, mean(e3r), floor(median(e3r)));
fprintf('-----\n');

end