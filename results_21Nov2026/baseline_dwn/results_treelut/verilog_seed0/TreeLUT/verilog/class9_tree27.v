module class9_tree27(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[238]) & i[195]) | (i[84] & (~i[195])) | ((~i[279]) & i[238] & i[195]) | (i[151] & (~i[84]) & (~i[195])) | (i[401] & i[254] & i[279] & i[238] & i[195]) | ((~i[401]) & i[254] & i[279] & i[238] & i[195]) | (i[225] & (~i[254]) & i[279] & i[238] & i[195]) | ((~i[225]) & (~i[254]) & i[279] & i[238] & i[195]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
