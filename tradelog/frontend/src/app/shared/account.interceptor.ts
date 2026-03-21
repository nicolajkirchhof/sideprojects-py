import { HttpInterceptorFn } from '@angular/common/http';

const STORAGE_KEY = 'tradelog-selected-account-id';

export const accountInterceptor: HttpInterceptorFn = (req, next) => {
  const accountId = localStorage.getItem(STORAGE_KEY);
  if (accountId) {
    const cloned = req.clone({
      setHeaders: { 'X-Account-Id': accountId },
    });
    return next(cloned);
  }
  return next(req);
};
