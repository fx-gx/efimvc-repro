function check_proteinfold(rateTag, m, alpha, beta, nreps)
% Run the OFFICIAL MATLAB code for a single EFIMVC configuration.
%
% Everything below is copied verbatim from official_code/run.m, restricted to
% one missing-rate file and one (m, alpha, beta) triple, so that the official
% pipeline can be compared against the Python port.  The only changes are:
%   * `parfor` -> `for` (Octave runs parfor as a normal loop anyway)
%   * `tic/toc` bookkeeping removed
%   * `RandStream` reset wrapped in try/catch (not needed for the mean)
%
% usage:  octave-cli --eval "pkg load optim statistics; check_proteinfold('0.9',81,0.001,0.001,20)"

  addpath(genpath('official_code'));

  datadir = 'official_code/Data/';
  datafile = [datadir, 'proteinFold_Per', rateTag, '.mat'];
  load(datafile);

  gt = truelabel{1};
  k  = length(unique(gt));

  [X1, ind] = findindex(data, index);
  numview = length(X1);
  n = size(X1{1},2);

  lambda1 = m; lambda2 = alpha; lambda3 = beta;

  for iv = 1:numview
      Z{iv} = client_pretrain(X1{iv}, lambda1, lambda2, ind(:,iv));
  end
  Zall = server_pretrain(Z, ind, lambda1);

  flag = 1; iter = 0; maxIter = 5; obj = [];
  for iv = 1:numview
      P{iv} = diag(ones(lambda1,1));
  end
  while flag
      iter = iter + 1;
      for iv = 1:numview
          [Z{iv}, S{iv}, obj_client(iv)] = client_train(X1{iv}, Z{iv}, lambda1, lambda2, ind(:,iv), Zall, P{iv});
      end
      [Zall, P, obj_server] = server_train(P, Zall, Z, S, lambda3, ind);
      obj(iter) = sum(obj_client) + obj_server;
      if (iter>1) && (abs((obj(iter-1)-obj(iter))/obj(iter-1))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
          [UU,~,~] = mySVD(Zall', k);
          flag = 0;
      end
  end

  F = UU ./ repmat(sqrt(sum(UU .^ 2, 2)), 1, size(UU,2));

  try
      stream = RandStream.getGlobalStream; reset(stream);
  catch
  end

  res = zeros(nreps, 8);
  for rep = 1:nreps
      pY = kmeans(F, k, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
      res(rep,:) = Clustering8Measure(gt, pY);
  end
  mres = mean(res);

  printf('OBJ %s\n', mat2str(obj, 6));
  printf('RESULT rate=%s m=%d alpha=%g beta=%g iters=%d ACC=%.2f NMI=%.2f Purity=%.2f Fscore=%.2f\n', ...
         rateTag, m, alpha, beta, iter, 100*mres(1), 100*mres(2), 100*mres(3), 100*mres(4));
end
