# �����M����DNN  

## �T�v  

�摜�F���ł�VGG16�Ȃǎ��O�w�K�������̂𗘗p�ł��邪�A�����F���p�r�ł͏��Ȃ��B 
�����ŁA�����F���G���W��Julius�̃f�B�N�e�[�V�����L�b�g�Ɋ܂܂��DNN�𗘗p���邽�߂� 
������FBANK_D_A_Z���v�Z����python������Ă݂��B 
  
## �g����  
### 1.DNN��model�̏���  
Julius�̃f�B�N�e�[�V�����L�b�gversion 4.4���_�E�����[�h����B<http://julius.osdn.jp/index.php?q=dictation-kit.html>  
model/dnn�ȉ���W�J����B 
  

### 2.�e�v���O�����̐���  
  
- get_fbank.py  16KHz�T���v�����O��wav�t�@�C����ǂݍ���œ�����FBANK_D_A_Z���v�Z����N���X�B
- cmvn_class.py  ���ϒl�E���U�̐��K��������N���X�B
- dnn_class.py  numpy��DNN���v�Z����N���X�B
- chainer_dnn_class.py �f�B�[�v���[�j���O�̃t���[�����[�N��chainer��DNN���v�Z����N���X�B�w�K�͖��Ή��B
- main0.py 16KHz�T���v�����O��wav�t�@�C����ǂݍ���� numpy��DNN���v�Z����܂ł�main�v���O�����̃T���v���B
- mainc.py 16KHz�T���v�����O��wav�t�@�C����ǂݍ���� chainer��DNN���v�Z����܂ł�main�v���O�����̃T���v���B
- bin/common/dnnclient.py dnn�v�Z�̓��o�̓f�[�^��npy�t�@�C���ŏ����o���ύX���������́B



## ����  
DNN�̌v�Z�o�͂�HMM�̉B���Ԃ̊m���Ȃ̂ŁA���̂܂܂ł͔F���Ɏg���Ȃ��B   
�v�Z�o�͂͏�Ԃ̗D��x�Ŋ�����LOG10�����l�ɂȂ��Ă���B  
  
> HMM�͎���3��Ԃ�LR�^�ŁC4,874�̏�Ԃ���Ȃ��ԋ��L���f���ł���D 
> ��Ԋm����DNN�ɂ���ė^������D 
  
Julius(C����float)��python�̐��l�v�Z�̐��x�������ł͂Ȃ��̂ŁA�v�Z���ʂ͊��S�ɂ͈�v���Ȃ��B 
(���悻1.0E-5�I�[�_�[���x�̍����o��悤���B) 

## ���C�Z���X  
�ȉ��̃��C�Z���X�����Q�Ƃ̂��ƁB   
LICENSE-Julius Dictation Kit.txt  
LICENSE-Julius.txt  
LICENSE-PyHTK  





