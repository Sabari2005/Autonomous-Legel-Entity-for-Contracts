import React from 'react';
import styled from 'styled-components';

const ChatProcessLoader = ({ width = "30px", height = "30px" }) => {
  return (
    <StyledWrapper width={width} height={height}>
      <div className="loader" />
    </StyledWrapper>
  );
};

const StyledWrapper = styled.div`
  .loader {
    border: 4px solid rgba(0, 0, 0, .1);
    border-left-color: transparent;
    border-radius: 50%;
  }

  .loader {
    border: 4px solid rgba(0, 0, 0, .1);
    border-left-color: transparent;
    width: ${props => props.width};
    height: ${props => props.height};
  }

  .loader {
    border: 4px solid white;
    border-left-color: transparent;
    width: ${props => props.width};
    height: ${props => props.height};
    animation: spin89345 1s linear infinite;
  }

  @keyframes spin89345 {
    0% {
      transform: rotate(0deg);
    }

    100% {
      transform: rotate(360deg);
    }
  }`;

export default ChatProcessLoader;
