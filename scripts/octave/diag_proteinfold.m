function diag_proteinfold(rateTag, m, alpha)
% Print intermediate quantities of the OFFICIAL code right after initialisation
% (client_pretrain + server_pretrain), so they can be diffed against the Python
% port without paying for the outer loop.
%
% usage: octave-cli --eval "pkg load optim statistics; diag_proteinfold('0.1',27,0.001)"

  addpath(genpath('official_code'));
  load(['official_code/Data/proteinFold_Per', rateTag, '.mat']);
  gt = truelabel{1};
  k = length(unique(gt));
  [X1, ind] = findindex(data, index);
  numview = length(X1);

  xn2 = zeros(1, numview);
  for iv = 1:numview
      xn2(iv) = sum(sum(X1{iv}.^2));
  end
  printf('n=%d numview=%d k=%d\n', size(X1{1},2), numview, k);
  printf('present_per_view=%s\n', mat2str(sum(ind,1)));
  printf('Xnorm2_per_view=%s\n', mat2str(xn2));

  for iv = 1:numview
      Z{iv} = client_pretrain(X1{iv}, m, alpha, ind(:,iv));
  end
  Zall = server_pretrain(Z, ind, m);

  tot_fit = 0; tot_znorm = 0;
  for iv = 1:numview
      X = X1{iv}; Zv = Z{iv};
      A = (X*Zv') * pinv(Zv*Zv');
      tot_fit   = tot_fit   + sum(sum((X - A*Zv).^2));
      tot_znorm = tot_znorm + sum(sum(Zv.^2));
  end
  P = cell(1,numview);
  for iv = 1:numview, P{iv} = diag(ones(m,1)); end
  tot_align = 0;
  for iv = 1:numview
      tot_align = tot_align + sum(sum((P{iv}*Zall.*repmat(ind(:,iv)',m,1) - Z{iv}).^2));
  end

  printf('AFTER_PRETRAIN fit=%.4f znorm2=%.4f align=%.4f Zall_norm2=%.4f\n', ...
         tot_fit, tot_znorm, tot_align, sum(Zall(:).^2));
  printf('Z1_present_sum=%s\n', mat2str(sum(Z{1},1), 8));
  printf('Z1_col0_present %s\n', mat2str(full(Z{1}(:,find(ind(:,1)==1,1))), 8));
  printf('Zall_col0 %s\n', mat2str(full(Zall(:,find(ind(:,1)==1,1))), 8));
  printf('Z_norm2_per_view=%s\n', mat2str(cellfun(@(z) sum(z(:).^2), Z)));
end
