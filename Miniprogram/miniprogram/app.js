//app.js
App({

  /**
   * 当小程序初始化完成时，会触发 onLaunch（全局只触发一次）
   */
  onLaunch: function () {
      wx.cloud.init({
        env: 'charlie-9mgmr',
      })
      const userInfo=wx.getStorageInfoSync('userInfo')
      if(userInfo){
        this.globalData.userInfo=userInfo
      }
    //   //openid 
    //   wx.cloud.callFunction({
    //     name: 'login',
    //     data: {},
    //     success: res => {
    //       console.log("云函数 [login] user openid: ", res.result.openid)
    //       this.globalData.openid = res.result.openid;
    //     },
    //     fail: res => {
    //       console.log("云函数 [login] 调用失败", err)
    //     }
    //   })
    //   this.userInfo=userInfo

    
  //}
  },

  /**
   * 当小程序启动，或从后台进入前台显示，会触发 onShow
   */
  onShow: function (options) {
    
  },

  /**
   * 当小程序从前台进入后台，会触发 onHide
   */
  onHide: function () {
    
  },

  /**
   * 当小程序发生脚本错误，或者 api 调用失败时，会触发 onError 并带上错误信息
   */
  onError: function (msg) {
    
  },
  globalData:{
    userInfo:null,
  }
})
