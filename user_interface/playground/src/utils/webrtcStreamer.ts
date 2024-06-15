import Player from 'xgplayer';

interface StreamerRTCPeerConnection extends RTCPeerConnection {
  peerId?: number;
}

export class WebRtcStreamer {
  public currentVideoURL: string = '';
  public xgplayer: Player;
  public videoElement: HTMLVideoElement;
  public peerConnection: StreamerRTCPeerConnection | null = null;
  public peerConnectionConfig: RTCConfiguration | null = null;
  public offerOptions: RTCOfferOptions = {
    offerToReceiveAudio: false,
    offerToReceiveVideo: true,
  };
  public iceServers: RTCIceServer[] = [];
  public iceCandiates: RTCIceCandidate[] = [];
  /*
   * Interface with WebRTC-streamer API
   * @constructor
   * @param {string} videoElement - id of the video element tag
   * @param {string} streamerURL -  url of webrtc-streamer
   */
  constructor(
    xgplayer: Player,
    public streamerURL: string,
  ) {
    this.xgplayer = xgplayer;
    this.videoElement = xgplayer.video as HTMLVideoElement;
    // if (typeof videoElement === 'string')
    //   this.videoElement = document.getElementById(
    //     videoElement,
    //   ) as HTMLVideoElement;
    // else this.videoElement = videoElement;
  }
  // 内部使用打印fetch请求错误信息函数
  _handleHttpErrors(response: Response) {
    if (!response.ok) {
      throw Error(response.statusText);
    }
    return response;
  }
  /*
   * AJAX callback for Error
   */
  onError(status: string) {
    console.log('onError:' + status);
  }

  /*
   * Disconnect a WebRTC Stream and clear videoElement source
   */
  disconnect() {
    if (this.videoElement.srcObject) {
      (this.videoElement.srcObject as MediaStream)
        .getTracks()
        .forEach((track) => {
          track.stop();
          (this.videoElement.srcObject as MediaStream).removeTrack(track);
        });
    }
    if (this.peerConnection) {
      fetch(
        this.streamerURL + '/api/hangup?peerid=' + this.peerConnection.peerId,
      )
        .then(this._handleHttpErrors)
        .catch((error) => this.onError('hangup ' + error));

      try {
        this.peerConnection.close();
      } catch (e) {
        console.log('Failure close peer connection:' + e);
      }
      this.peerConnection = null;
    }
  }
  /*
   * Connect a WebRTC Stream to videoElement
   * @param {string} videourl - id of WebRTC video stream
   * @param {string} options  -  options of WebRTC call
   * @param {string} prefmime -  prefered mime-prefered codec
   */
  connect(videourl: string, options?: string, prefmime?: string) {
    this.disconnect();

    // getIceServers is not already received
    if (!this.iceServers) {
      console.log('Get IceServers');

      fetch(this.streamerURL + '/api/getIceServers')
        .then(this._handleHttpErrors)
        .then((response) => response.json())
        .then((response: RTCIceServer[] | undefined) =>
          this.onReceiveGetIceServers(response, videourl, options, prefmime),
        )
        .catch((error) => this.onError('getIceServers ' + error));
    } else {
      this.onReceiveGetIceServers(this.iceServers, videourl, options, prefmime);
    }
  }
  /*
   * GetIceServers callback
   */
  onReceiveGetIceServers(
    iceServers: RTCIceServer[] | undefined,
    videourl: string,
    options?: string,
    prefmime?: string,
  ) {
    this.iceServers = iceServers ?? [];
    this.peerConnectionConfig = { iceServers: this.iceServers };
    try {
      this.createPeerConnection();

      this.currentVideoURL = videourl;

      let callurl =
        this.streamerURL +
        '/api/call?peerid=' +
        this.peerConnection?.peerId +
        '&url=' +
        encodeURIComponent(videourl);

      if (options) {
        callurl += '&options=' + encodeURIComponent(options);
      }

      // clear early candidates
      this.iceCandiates.length = 0;

      // create Offer
      this.peerConnection?.createOffer(this.offerOptions).then(
        (sessionDescription) => {
          console.log('Create offer:' + JSON.stringify(sessionDescription));

          if (prefmime != undefined) {
            //set prefered codec
            const [prefkind] = prefmime.split('/');
            const codecs = RTCRtpReceiver?.getCapabilities(prefkind)?.codecs;
            console.log(`codecs:${JSON.stringify(codecs)}`);
            const preferedCodecs = codecs?.filter(
              (codec) => codec.mimeType === prefmime,
            );

            console.log(`preferedCodecs:${JSON.stringify(preferedCodecs)}`);
            this.peerConnection!.getTransceivers()
              .filter(
                (transceiver) => transceiver.receiver.track.kind === prefkind,
              )
              .forEach((tcvr) => {
                if (tcvr.setCodecPreferences != undefined) {
                  tcvr.setCodecPreferences(
                    preferedCodecs as RTCRtpCodecCapability[],
                  );
                }
              });
          }

          this.peerConnection?.setLocalDescription(sessionDescription).then(
            () => {
              fetch(callurl, {
                method: 'POST',
                body: JSON.stringify(sessionDescription),
              })
                .then(this._handleHttpErrors)
                .then((response) => response.json())
                .catch((error) => this.onError('call ' + error))
                .then((response) => this.onReceiveCall(response))
                .catch((error) => this.onError('call ' + error));
            },
            (error) => {
              console.log('setLocalDescription error:' + JSON.stringify(error));
            },
          );
        },
        (error) => {
          alert('Create offer error:' + JSON.stringify(error));
        },
      );
    } catch (e) {
      this.disconnect();
      alert('connect error: ' + e);
    }
  }

  /*
   * create RTCPeerConnection
   */
  createPeerConnection() {
    console.log(
      'createPeerConnection  config: ' +
        JSON.stringify(this.peerConnectionConfig),
    );
    this.peerConnection = new RTCPeerConnection(this.peerConnectionConfig!);
    const peerConnection = this.peerConnection;
    peerConnection.peerId = Math.random();

    peerConnection.onicecandidate = (evt) => this.onIceCandidate(evt);
    peerConnection.ontrack = (evt) => this.onAddStream(evt);
    peerConnection.oniceconnectionstatechange = () => {
      console.log(
        'oniceconnectionstatechange  state: ' +
          peerConnection.iceConnectionState,
      );
      if (this.videoElement) {
        if (peerConnection.iceConnectionState === 'connected') {
          this.videoElement.style.opacity = '1.0';
        } else if (peerConnection.iceConnectionState === 'disconnected') {
          this.videoElement.style.opacity = '0.25';
        } else if (
          peerConnection.iceConnectionState === 'failed' ||
          peerConnection.iceConnectionState === 'closed'
        ) {
          this.videoElement.style.opacity = '0.5';
        } else if (peerConnection.iceConnectionState === 'new') {
          this.getIceCandidate();
        }
      }
    };
    peerConnection.ondatachannel = function (evt) {
      console.log('remote datachannel created:' + JSON.stringify(evt));

      evt.channel.onopen = function () {
        console.log('remote datachannel open');
        this.send('remote channel openned');
      };
      evt.channel.onmessage = function (event) {
        console.log('remote datachannel recv:' + JSON.stringify(event.data));
      };
    };
    peerConnection.onicegatheringstatechange = function () {
      if (peerConnection.iceGatheringState === 'complete') {
        const recvs = peerConnection.getReceivers();

        recvs.forEach((recv) => {
          if (recv.track && recv.track.kind === 'video') {
            console.log(
              'codecs:' + JSON.stringify(recv.getParameters().codecs),
            );
          }
        });
      }
    };

    try {
      const dataChannel = peerConnection.createDataChannel('ClientDataChannel');
      dataChannel.onopen = function () {
        console.log('local datachannel open');
        this.send('local channel openned');
      };
      dataChannel.onmessage = function (evt) {
        console.log('local datachannel recv:' + JSON.stringify(evt.data));
      };
    } catch (e) {
      console.log('Cannor create datachannel error: ' + e);
    }

    console.log(
      'Created RTCPeerConnnection with config: ' +
        JSON.stringify(this.peerConnectionConfig),
    );
    return peerConnection;
  }

  /*
   * RTCPeerConnection IceCandidate callback
   */
  onIceCandidate(event: RTCPeerConnectionIceEvent) {
    if (event.candidate) {
      if (this.peerConnection!.currentRemoteDescription) {
        this.addIceCandidate(this.peerConnection!.peerId!, event.candidate);
      } else {
        this.iceCandiates.push(event.candidate);
      }
    } else {
      console.log('End of candidates.');
    }
  }

  addIceCandidate(peerId: number, candidate: RTCIceCandidate) {
    fetch(this.streamerURL + '/api/addIceCandidate?peerid=' + peerId, {
      method: 'POST',
      body: JSON.stringify(candidate),
    })
      .then(this._handleHttpErrors)
      .then((response) => response.json())
      .then((response) => {
        console.log('addIceCandidate ok:' + response);
      })
      .catch((error) => this.onError('addIceCandidate ' + error));
  }

  /*
   * RTCPeerConnection AddTrack callback
   */
  onAddStream(event: RTCTrackEvent) {
    console.log('Remote track added:' + JSON.stringify(event));

    this.videoElement.srcObject = event.streams[0];
    const promise = this.videoElement.play();
    if (promise !== undefined) {
      promise.catch((error) => {
        console.warn('error:' + error);
      });
    }
  }

  /*
   * AJAX /getIceCandidate callback
   */

  getIceCandidate() {
    fetch(
      this.streamerURL +
        '/api/getIceCandidate?peerid=' +
        this.peerConnection?.peerId,
    )
      .then(this._handleHttpErrors)
      .then((response) => response.json())
      .then((response: RTCIceCandidateInit[]) =>
        this.onReceiveCandidate(response),
      )
      .catch((error) => this.onError('getIceCandidate ' + error));
  }

  onReceiveCandidate(dataJson: RTCIceCandidateInit[] | undefined) {
    console.log('candidate: ' + JSON.stringify(dataJson));
    if (dataJson) {
      for (let i = 0; i < dataJson.length; i++) {
        const candidate = new RTCIceCandidate(dataJson[i]);

        console.log('Adding ICE candidate :' + JSON.stringify(candidate));
        this.peerConnection?.addIceCandidate(candidate).then(
          () => {
            console.log('addIceCandidate OK');
          },
          (error) => {
            console.log('addIceCandidate error:' + JSON.stringify(error));
          },
        );
      }
      this.peerConnection?.addIceCandidate();
    }
  }

  /*
   * AJAX /call callback
   */
  onReceiveCall(dataJson: RTCSessionDescriptionInit) {
    console.log('offer: ' + JSON.stringify(dataJson));
    const descr = new RTCSessionDescription(dataJson);
    this.peerConnection?.setRemoteDescription(descr).then(
      () => {
        console.log('setRemoteDescription ok');
        while (this.iceCandiates.length) {
          const candidate = this.iceCandiates.shift();
          this.addIceCandidate(this.peerConnection!.peerId!, candidate!);
        }

        this.getIceCandidate();
      },
      (error) => {
        console.log('setRemoteDescription error:' + JSON.stringify(error));
      },
    );
  }
}
