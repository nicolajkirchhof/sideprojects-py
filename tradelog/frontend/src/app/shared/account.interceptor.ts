import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AccountsService } from '../accounts/accounts.service';

export const accountInterceptor: HttpInterceptorFn = (req, next) => {
  const accountId = inject(AccountsService).selectedAccountId();
  if (accountId) {
    const cloned = req.clone({
      setHeaders: { 'X-Account-Id': accountId.toString() },
    });
    return next(cloned);
  }
  return next(req);
};
