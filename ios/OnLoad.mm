#import <Foundation/Foundation.h>
#import "PureCppImpl.h"
#import <ReactCommon/CxxTurboModuleUtils.h>


@interface OnLoad : NSObject
@end

@implementation OnLoad

using namespace facebook::react;

+ (void) load {
  registerCxxModuleToGlobalModuleMap(
    std::string(PureCppImpl::kModuleName),
    [](std::shared_ptr<CallInvoker> jsInvoker) {
      return std::make_shared<PureCppImpl>(jsInvoker);
    }
  );
}

@end